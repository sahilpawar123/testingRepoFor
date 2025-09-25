from logging import Logger
from typing import Any
import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import sys
from dateutil import parser
import re
from config.constants import TRANSACTION_TYPES, CATEGORIES
from services.google_sheets import get_sheets_service
from utils.logging_utils import setup_logging

log: Logger = setup_logging("expense_tracker")

# Load environment variables
load_dotenv()
log.info("✨ Environment variables loaded")

st.set_page_config (layout='wide')

# Configure Gemini AI
@st.cache_resource
def get_gemini_model() -> Any:
    """Cache Gemini AI configuration"""
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY')) # type: ignore
        model: Any = genai.GenerativeModel('gemini-1.5-flash') # type: ignore
        log.info("🤖 Gemini AI configured successfully")
        return model
    except Exception as e:
        log.error(f"❌ Failed to configure Gemini AI: {str(e)}")
        raise


# Replace the direct configuration with cached versions
try:
    model = get_gemini_model()
    service = get_sheets_service()
    SHEET_ID: str | None = os.getenv('GOOGLE_SHEET_ID')
    log.info("📊 Google Sheets API connected successfully")
except Exception as e:
    log.error(f"❌ Failed to connect to Google Sheets: {str(e)}")
    log.error(f"❌ Failed to initialize services: {str(e)}")
    sys.exit(1)


@st.cache_data(ttl=300)
def get_categories() -> dict[str, dict[str, list[str]]]:
    """Cache the categories dictionary to prevent reloading"""
    return CATEGORIES

@st.cache_data
def get_transaction_types() -> list[str]:
    """Cache the transaction types to prevent reloading"""
    return TRANSACTION_TYPES

def init_session_state() -> None:
    """
    Initialize Streamlit session state variables with default values.
    Sets up necessary state variables for the application.
    """
    defaults: dict[str, Any] = {
        'messages': [],
        'save_clicked': False,
        'current_amount': None,
        'current_type': None,
        'current_category': None,
        'current_subcategory': None,
        'form_submitted': False,
        'show_analytics': False,  # New state variable for analytics
        'current_transaction': None,  # New state variable for current transaction
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def parse_date_from_text(text: str) -> datetime:
    """
    Extract and parse date from input text.
    
    Args:
        text (str): Input text containing date information
        
    Returns:
        str: Parsed date in YYYY-MM-DD format
    """
    current_date: datetime = datetime.now()
    try:
        text = text.lower()
        
        relative_dates: dict[str, datetime] = {
            'today': current_date,
            'yesterday': current_date - timedelta(days=1),
            'tomorrow': current_date + timedelta(days=1),
            'day before yesterday': current_date - timedelta(days=2),
        }

        for phrase, date in relative_dates.items():
            if phrase in text:
                return date

        last_pattern: str = r'last (\d+) (day|week|month)s?'
        match: re.Match[str] | None = re.search(last_pattern, text)
        if match:
            number: int = int(match.group(1))
            unit: str | Any = match.group(2)
            if unit == 'day':
                return current_date - timedelta(days=number)
            elif unit == 'week':
                return current_date - timedelta(weeks=number)
            elif unit == 'month':
                return current_date - timedelta(days=number * 30)

        next_pattern = r'next (\d+) (day|week|month)s?'
        match = re.search(next_pattern, text)
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            if unit == 'day':
                return current_date + timedelta(days=number)
            elif unit == 'week':
                return current_date + timedelta(weeks=number)
            elif unit == 'month':
                return current_date + timedelta(days=number * 30)

        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        match = re.search(date_pattern, text)
        if match:
            return parser.parse(match.group())
        words: list[str] = text.split()
        for i in range(len(words)-2):
            possible_date: str = ' '.join(words[i:i+3])
            try:
                return parser.parse(possible_date)
            except Exception as e:
                log.error(f"❌ Failed to parse date from text: {str(e)}")
                continue
        
        return current_date
    
    except Exception as e:
        log.warning(f"Failed to parse date from text, using current date. Error: {str(e)}")
        return current_date

def test_sheet_access() -> bool:
    """
    Test Google Sheets API connection.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Test write access by appending to the last row instead of clearing
        test_values: list[list[str]] = [['TEST', 'TEST', 'TEST', 'TEST', 'TEST', 'TEST']]
        result: Any = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range='Expenses',
            valueInputOption='RAW',
            body={'values': test_values}
        ).execute()
        
        # Get the range that was just written
        updated_range:str = result['updates']['updatedRange']
        
        # Only clear the test row we just added
        service.spreadsheets().values().clear(
            spreadsheetId=SHEET_ID,
            range=updated_range,
            body={}
        ).execute()
        
        log.info("✅ Sheet access test successful")
        return True
    except Exception as e:
        log.error(f"❌ Sheet access test failed: {str(e)}")
        return False

def initialize_sheet() -> None:
    try:
        # Create sheets if they don't exist
        sheet_metadata: Any = service.spreadsheets().get(spreadsheetId=SHEET_ID).execute()
        sheets: list[Any] = sheet_metadata.get('sheets', '')
        existing_sheets: set[Any] = {s.get("properties", {}).get("title") for s in sheets}
        
        # Initialize Expenses sheet
        if 'Expenses' not in existing_sheets:
            log.info("Creating new Expenses sheet...")
            body: dict[str, Any] = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': 'Expenses'
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=SHEET_ID,
                body=body
            ).execute()
            
            headers: list[list[str]] = [['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description']]
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range='Expenses!A1:F1',
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
        
        # Initialize Pending sheet
        if 'Pending' not in existing_sheets:
            log.info("Creating new Pending sheet...")
            body: dict[str, Any] = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': 'Pending'
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=SHEET_ID,
                body=body
            ).execute()
            
            headers: list[list[str]] = [['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status']]
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range='Pending!A1:G1',
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
        
        # Test sheet access
        if not test_sheet_access():
            raise Exception("Failed to verify sheet access")
            
        log.info("✨ Sheets initialized and verified")
    except Exception as e:
        log.error(f"❌ Failed to initialize sheets: {str(e)}")
        raise

def add_transaction_to_sheet(date: str, amount: float, trans_type: str, 
                           category: str, subcategory: str, description: str) -> bool:
    """
    Add a new transaction to Google Sheet.
    
    Args:
        date (str): Transaction date in YYYY-MM-DD format
        amount (float): Transaction amount
        trans_type (str): Type of transaction (Income/Expense)
        category (str): Transaction category
        subcategory (str): Transaction subcategory
        description (str): Transaction description
        
    Returns:
        bool: True if transaction added successfully, False otherwise
    """
    try:
        log.info(f"Starting transaction save: {date}, {amount}, {trans_type}, {category}, {subcategory}, {description}")
        
        # Format the date if it's a datetime object
        date_str:Any = date
        
        # Ensure amount is a string
        amount_str: str = str(float(amount))
        
        # Prepare the values
        values: list[list[str]] = [[str(date_str), amount_str, trans_type, category, subcategory, description]]
        
        # Changed range to 'Expenses' to let Google Sheets determine the next empty row
        result: Any = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range='Expenses',  # Changed from 'Expenses!A2:F2' to just 'Expenses'
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body={'values': values}
        ).execute()
        
        log.info(f"✅ Transaction saved successfully: {result}")
        return True
        
    except Exception as e:
        log.error(f"❌ Failed to save transaction: {str(e)}")
        return False

@st.cache_data(ttl=300)  
def get_transactions_data() -> pd.DataFrame:
    """
    Fetch and process all transactions from Google Sheet.
    
    Returns:
        pd.DataFrame: Processed transactions data
    """
    try:
        log.debug("Fetching transactions data from Google Sheets")
        result: Any = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Expenses!A1:F'
        ).execute()
        
        values: list[list[str]] = result.get('values', [])
        if not values:
            log.warning("No transaction data found in sheet")
            return pd.DataFrame(columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description'])
        
        log.info(f"📈 Retrieved {len(values)-1} transaction records")
        return pd.DataFrame(values[1:], columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description'])
    except Exception as e:
        log.error(f"❌ Failed to fetch transactions data: {str(e)}")
        raise

def validate_amount(amount_str: str) -> float:
    """
    Validate and convert amount string to float.
    
    Args:
        amount_str: String representation of amount
        
    Returns:
        float: Validated amount
        
    Raises:
        ValueError: If amount is invalid
    """
    try:
        amount = float(amount_str)
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return amount
    except ValueError as e:
        log.error(f"❌ Invalid amount: {amount_str}")
        raise ValueError(f"Invalid amount: {amount_str}") from e

def classify_transaction_type(text: str, model: Any) -> dict[str, Any]:
    """
    Use Gemini to classify the type of transaction.
    """
    try:
        log.info("🔍 Starting transaction classification")
        log.debug(f"Input text: {text}")
        
        chat = model.start_chat(history=[])
        prompt = f"""
        Classify this transaction: '{text}'
        
        VERY IMPORTANT CLASSIFICATION RULES:
        1. If text contains "received pending" or "got pending" or "collected pending":
           -> MUST classify as PENDING_RECEIVED
           Example: "received pending money of 1275" -> PENDING_RECEIVED
           Example: "got pending payment of 500" -> PENDING_RECEIVED
        
        2. If text indicates future receipt WITHOUT "pending":
           -> classify as PENDING_TO_RECEIVE
           Example: "will receive 1000 next week" -> PENDING_TO_RECEIVE
        
        3. If text indicates future payment:
           -> classify as PENDING_TO_PAY
           Example: "need to pay 500 tomorrow" -> PENDING_TO_PAY
        
        4. If text indicates immediate expense:
           -> classify as EXPENSE_NORMAL
           Example: "spent 100 on food" -> EXPENSE_NORMAL
        
        5. If text indicates immediate income WITHOUT "pending":
           -> classify as INCOME_NORMAL
           Example: "got salary 5000" -> INCOME_NORMAL
        
        IMPORTANT: For any text containing "received pending", "got pending", or "collected pending",
        you MUST classify it as PENDING_RECEIVED, regardless of other words in the text.
        
        Respond in this format ONLY:
        type: <PENDING_RECEIVED/PENDING_TO_RECEIVE/PENDING_TO_PAY/EXPENSE_NORMAL/INCOME_NORMAL>
        amount: <positive number only>
        description: <brief description>
        """
        
        log.debug("🤖 Sending classification prompt to Gemini")
        response = chat.send_message(prompt)
        lines = response.text.strip().split('\n')
        result: dict[str, Any] = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        
        # Double-check classification for pending received
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in ['received pending', 'got pending', 'collected pending']):
            if result.get('type') != 'PENDING_RECEIVED':
                log.warning(f"⚠️ Correcting misclassified pending received transaction: {result.get('type')} -> PENDING_RECEIVED")
                result['type'] = 'PENDING_RECEIVED'
        
        # Validate required fields
        required_fields = ['type', 'amount', 'description']
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
        # Validate transaction type
        valid_types = ['EXPENSE_NORMAL', 'INCOME_NORMAL', 'PENDING_TO_RECEIVE', 
                      'PENDING_TO_PAY', 'PENDING_RECEIVED', 'PENDING_PAID']
        if result['type'] not in valid_types:
            raise ValueError(f"Invalid transaction type: {result['type']}")
            
        # Validate amount
        result['amount'] = str(validate_amount(result['amount']))
        
        log.info(f"📋 Transaction classified as: {result.get('type', 'UNKNOWN')}")
        log.debug(f"Classification details: {result}")
        return result
    except Exception as e:
        log.error(f"❌ Failed to classify transaction: {str(e)}")
        raise

def handle_received_pending_transaction(amount: float, description: str) -> tuple[bool, dict[str, Any] | None]:
    """
    Handle a pending transaction that has been received.
    """
    try:
        if amount <= 0:
            raise ValueError("Amount must be positive")
            
        log.info(f"💫 Processing received pending transaction: amount={amount}")
        
        # First check if this transaction was already processed today
        log.debug("Checking for existing received transactions today")
        today = datetime.now().strftime('%Y-%m-%d')
        
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Expenses!A:F'
        ).execute()
        
        values = result.get('values', [])
        if values and len(values) > 1:  # Check if we have data beyond header
            for row in values[1:]:  # Skip header
                if (len(row) >= 6 and 
                    row[0] == today and 
                    abs(float(row[1]) - amount) < 0.01 and
                    row[2] == 'Income' and
                    row[3] == 'Other' and
                    row[4] == 'Pending Received' and
                    'received pending' in row[5].lower()):
                    log.warning("⚠️ This pending transaction was already processed today")
                    return False, None
        
        # Now check pending transactions
        log.debug("Searching for matching pending transaction")
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Pending!A:G'
        ).execute()
        
        values = result.get('values', [])
        if not values:
            log.warning("❗ No pending transactions found in sheet")
            return False, None
            
        # Skip header row and find matching pending transaction
        matching_rows: list[int] = []
        
        # Validate sheet structure
        if len(values[0]) < 7:
            log.error("❌ Invalid sheet structure: missing required columns")
            return False, None
            
        # Start from index 1 to skip header row
        for i, row in enumerate(values[1:], start=1):
            try:
                if len(row) < 7:
                    log.warning(f"⚠️ Skipping row {i+1}: insufficient columns")
                    continue
                    
                row_amount = float(row[1])
                if (abs(row_amount - amount) < 0.01 and  # Use small epsilon for float comparison
                    row[6] == 'Pending' and 
                    row[2] == 'To Receive'):
                    matching_rows.append(i)
                    log.debug(f"Found potential match at row {i+1}: amount={row_amount}")
            except (ValueError, IndexError) as e:
                log.warning(f"⚠️ Error processing row {i+1}: {str(e)}")
                continue
        
        if len(matching_rows) > 1:
            log.warning(f"⚠️ Multiple matching pending transactions found for amount {amount}")
            # Use the most recent transaction if multiple matches
            row_index: int = matching_rows[-1]
            log.info(f"Selected most recent match at row {row_index+1}")
        elif len(matching_rows) == 1:
            row_index = matching_rows[0]
            log.info(f"✅ Found matching pending transaction at row {row_index+1}")
        else:
            log.warning(f"❗ No matching pending transaction found for amount {amount}")
            return False, None
            
        # Update status to Received
        log.debug(f"Updating status to Received for row {row_index+1}")
        range_name = f'Pending!G{row_index + 1}'
        try:
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range=range_name,
                valueInputOption='RAW',
                body={'values': [['Received']]}
            ).execute()
        except Exception as e:
            log.error(f"❌ Failed to update pending transaction status: {str(e)}")
            return False, None
        
        # Get original transaction details
        original_row = values[row_index]
        original_date = original_row[0]
        original_description = original_row[4] if len(original_row) > 4 else ''
        
        # Create transaction info
        transaction_info = {
            'type': 'Income',
            'amount': str(amount),
            'category': 'Other',
            'subcategory': 'Pending Received',
            'description': f"Received pending payment ({original_date}): {original_description}",
            'date': today
        }
        
        # Add as new Income transaction
        log.debug("Creating new Income transaction")
        success = add_transaction_to_sheet(
            transaction_info['date'],
            amount,
            transaction_info['type'],
            transaction_info['category'],
            transaction_info['subcategory'],
            transaction_info['description']
        )
        
        if success:
            log.info("✨ Successfully processed received pending transaction")
        else:
            log.error("❌ Failed to create Income transaction")
        
        return success, transaction_info if success else None
        
    except Exception as e:
        log.error(f"❌ Failed to handle received pending transaction: {str(e)}")
        return False, None

def process_user_input(text: str) -> dict[str, Any]:
    """
    Process natural language input to extract transaction details.
    """
    try:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
            
        log.info("🎯 Starting transaction processing")
        log.debug(f"Processing input: {text}")
        
        # First, classify the transaction type
        log.debug("Step 1: Classifying transaction type")
        classification = classify_transaction_type(text, model)
        transaction_type = classification.get('type', '')
        
        try:
            amount = float(classification.get('amount', 0))
            if amount <= 0:
                raise ValueError("Amount must be positive")
        except ValueError as e:
            log.error(f"❌ Invalid amount in classification: {classification.get('amount')}")
            raise
        
        log.info(f"Transaction classified as: {transaction_type}")
        
        # Handle each type differently
        if transaction_type == 'PENDING_RECEIVED':
            log.info("🔄 Handling received pending transaction")
            success, transaction_info = handle_received_pending_transaction(amount, text)
            if success and transaction_info:
                log.info("✅ Successfully processed received pending transaction")
                # Mark transaction as auto-processed to skip form
                transaction_info['auto_processed'] = True
                return transaction_info
            else:
                log.warning("⚠️ Failed to process received pending transaction")
                raise ValueError("Failed to process received pending transaction")
        
        elif transaction_type == 'PENDING_PAID':
            log.info("💰 Handling paid pending transaction")
            # TODO: Implement handling paid pending payments
            raise NotImplementedError("Handling paid pending transactions is not implemented yet")
            
        # For other types, get detailed transaction info
        log.debug("Step 2: Getting detailed transaction info")
        chat = model.start_chat(history=[])
        prompt = f"""
        Extract transaction information from this text: '{text}'
        Transaction was classified as: {transaction_type}
        
        Based on the classification, follow these rules:
        
        1. For EXPENSE_NORMAL:
           -> Set type: "Expense"
           -> Choose category from: Food/Transportation/Housing/Entertainment/Shopping/Healthcare/Gift/Other
        
        2. For INCOME_NORMAL:
           -> Set type: "Income"
           -> Choose category from: Salary/Investment/Other
        
        3. For PENDING_TO_RECEIVE:
           -> Set type: "To Receive"
           -> Set category: "Pending Income"
        
        4. For PENDING_TO_PAY:
           -> Set type: "To Pay"
           -> Choose category from: Bills/Debt
        
        5. For PENDING_RECEIVED:
           -> Set type: "Income"
           -> Set category: "Other"
           -> Set subcategory: "Pending Received"
        
        6. For PENDING_PAID:
           -> Set type: "Expense"
           -> Use original pending payment category
        
        Respond in this EXACT format (include ALL fields):
        type: <Income/Expense/To Receive/To Pay>
        amount: <number only>
        category: <must match categories listed above>
        subcategory: <must match valid subcategories>
        description: <brief description>
        due_date: <YYYY-MM-DD format, ONLY for To Receive/To Pay>
        """
        
        log.debug("🤖 Sending detail extraction prompt to Gemini")
        response = chat.send_message(prompt)
        response_text: str = response.text
        lines: list[str] = response_text.strip().split('\n')
        extracted_info: dict[str, Any] = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                extracted_info[key.strip()] = value.strip().replace('"', '').replace("'", "")
        
        log.debug(f"Extracted transaction details: {extracted_info}")
        
        # Set current date as transaction date
        current_date: str = datetime.now().strftime('%Y-%m-%d')
        extracted_info['date'] = current_date
        
        # Handle relative dates in due_date
        if extracted_info.get('type') in ['To Receive', 'To Pay']:
            log.debug("Processing due date for pending transaction")
            if 'due_date' not in extracted_info or not extracted_info.get('due_date', '').strip():
                due_date: str = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                extracted_info['due_date'] = due_date
                log.debug(f"No due date provided, defaulting to: {due_date}")
            else:
                try:
                    parsed_date: datetime = parser.parse(str(extracted_info.get('due_date', '')))
                    extracted_info['due_date'] = parsed_date.strftime('%Y-%m-%d')
                    log.debug(f"Parsed due date: {extracted_info['due_date']}")
                except:
                    due_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    extracted_info['due_date'] = due_date
                    log.warning(f"Failed to parse due date, defaulting to: {due_date}")
        
        log.info("✅ Successfully processed transaction")
        log.debug(f"Final transaction info: {extracted_info}")
        return extracted_info
        
    except Exception as e:
        log.error(f"❌ Failed to process user input: {str(e)}", exc_info=True)
        raise

def show_analytics() -> None:
    """
    Display analytics dashboard with transaction visualizations.
    Shows pie charts and trends for income and expenses.
    """
    try:
        log.info("Generating financial analytics")
        df = get_transactions_data()
        
        if df.empty:
            st.info("No transactions recorded yet. Add some transactions to see analytics!")
            return
            
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce') # type: ignore
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # type: ignore
        
        # Calculate totals
        total_income = df[df['Type'] == 'Income']['Amount'].sum() # type: ignore 
        total_expenses = df[df['Type'] == 'Expense']['Amount'].sum() # type: ignore
        net_balance = total_income - total_expenses
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Income", f"Rs. {total_income:,.2f}", delta=None)
        with col2:
            st.metric("Total Expenses", f"Rs. {total_expenses:,.2f}", delta=None)
        with col3:
            st.metric("Net Balance", f"Rs. {net_balance:,.2f}", 
                     delta=f"Rs. {net_balance:,.2f}", 
                     delta_color="normal" if net_balance >= 0 else "inverse")
        
        if len(df) > 1:  # Only show charts if we have more than one transaction
            # Income vs Expenses over time
            df_grouped = df.groupby(['Date', 'Type'])['Amount'].sum().unstack(fill_value=0) # type: ignore
            fig_timeline = px.line(df_grouped,  # type: ignore
                                 title='Income vs Expenses Over Time',
                                 labels={'value': 'Amount (Rs. )', 'variable': 'Type'})
            st.plotly_chart(fig_timeline) # type: ignore
            
            # Category breakdown for both income and expenses
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Income Breakdown")
                income_df = df[df['Type'] == 'Income']
                if not income_df.empty:
                    fig_income = px.pie(income_df, values='Amount', names='Category',  # type: ignore
                                      title='Income by Category')
                    st.plotly_chart(fig_income) # type: ignore
                else:
                    st.info("No income transactions recorded yet.")
            
            with col2:
                st.subheader("Expense Breakdown")
                expense_df = df[df['Type'] == 'Expense']
                if not expense_df.empty:
                    fig_expense = px.pie(expense_df, values='Amount', names='Category',  # type: ignore
                                       title='Expenses by Category')
                    st.plotly_chart(fig_expense) # type: ignore
                else:
                    st.info("No expense transactions recorded yet.")
            
            # Monthly summary
            st.subheader("Monthly Summary")
            monthly_summary = df.groupby([df['Date'].dt.strftime('%Y-%m'), 'Type'])['Amount'].sum().unstack(fill_value=0) # type: ignore
            monthly_summary['Net'] = monthly_summary.get('Income', 0) - monthly_summary.get('Expense', 0) # type: ignore
            st.dataframe(monthly_summary.style.format("Rs. {:,.2f}")) # type: ignore
        
        log.info("✅ Analytics visualizations generated successfully")
    except Exception as e:
        log.error(f"❌ Failed to generate analytics: {str(e)}")
        st.error("Failed to generate analytics. Please try again later.")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_sheet_url() -> str:
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"

@st.cache_resource  # Cache for the entire session
def initialize_gemini() -> Any:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY')) # type: ignore
    return genai.GenerativeModel('gemini-1.5-flash') # type: ignore

@st.cache_data
def get_subcategories(trans_type: str, category: str) -> list[str]:
    return CATEGORIES[trans_type][category]

def on_save_click():
    st.session_state.save_clicked = True

def verify_sheet_setup() -> bool:
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Expenses!A1:F1'
        ).execute()
        
        values = result.get('values', [])
        expected_headers = ['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description']
        
        if not values or values[0] != expected_headers:
            # Reinitialize headers
            headers = [expected_headers]
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range='Expenses!A1:F1',
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
            log.info("Headers reinitialized")
            
        return True
    except Exception as e:
        log.error(f"Failed to verify sheet setup: {str(e)}")
        return False

def show_success_message(transaction_date: datetime | str, subcategory: str | None) -> None:
    """
    Display success message after transaction is saved.
    
    Args:
        transaction_date: Date of the transaction
        subcategory: Transaction subcategory, if applicable
    """
    emoji = "💰" if st.session_state.current_transaction['type'] == "Income" else "💸"
    confirmation_message = (
        f"{emoji} Transaction recorded:\n\n"
        f"Date: {transaction_date}\n"
        f"Amount: Rs. {float(st.session_state.current_transaction['amount']):,.2f}\n"
        f"Type: {st.session_state.current_transaction['type']}\n"
        f"Category: {st.session_state.current_transaction['category']}\n"
        f"Subcategory: {subcategory if subcategory else 'N/A'}"
    )
    st.success(confirmation_message)
    st.session_state.messages.append({"role": "assistant", "content": confirmation_message})
    log.info("✅ Transaction saved and analytics updated")

def show_transaction_form():
    """Separate function to handle transaction form display and processing"""
    extracted_info = st.session_state.current_transaction
    
    # Skip form for auto-processed transactions (like received pending)
    if extracted_info.get('auto_processed'):
        log.debug("Showing feedback for auto-processed transaction")
        
        # Show detailed success message
        st.success("✅ Transaction Processed Successfully")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Transaction Details:**")
            st.write(f"📅 Date: {extracted_info.get('date')}")
            st.write(f"💰 Amount: Rs. {float(extracted_info.get('amount', 0)):,.2f}")
            st.write(f"📝 Type: {extracted_info.get('type')}")
            
        with col2:
            st.write(f"🏷️ Category: {extracted_info.get('category')}")
            st.write(f"📑 Subcategory: {extracted_info.get('subcategory')}")
            st.write(f"📌 Description: {extracted_info.get('description')}")
            
        # Add a divider for visual separation
        st.divider()
        
        # Add a clear button
        if st.button("Clear Message", key="clear_feedback"):
            st.session_state.current_transaction = None
            st.rerun()
        return
    
    if 'amount' in extracted_info and 'type' in extracted_info:
        # Create form container
        form_container = st.container()
        
        with form_container:
            # Initialize form state
            if 'form_submitted' not in st.session_state:
                st.session_state.form_submitted = False
            
            with st.form(key="transaction_form"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if extracted_info['type'] in ['To Receive', 'To Pay']:
                        # For pending transactions
                        try:
                            # Try to parse the due date if it exists
                            if 'due_date' in extracted_info and extracted_info['due_date']:
                                default_due_date = datetime.strptime(extracted_info['due_date'], '%Y-%m-%d')
                            else:
                                # Default to 7 days from now
                                default_due_date = datetime.now() + timedelta(days=7)
                        except ValueError:
                            # If parsing fails, use 7 days from now
                            default_due_date = datetime.now() + timedelta(days=7)
                        
                        due_date = st.date_input(
                            "Due date",
                            value=default_due_date,
                            key="due_date"
                        )
                    else:
                        # For regular transactions
                        categories = get_categories()
                        subcategories = categories[extracted_info['type']][extracted_info['category']]
                        subcategory = st.selectbox(
                            "Select subcategory",
                            subcategories,
                            key="subcategory_select"
                        )
                    
                    default_date = datetime.strptime(extracted_info['date'], '%Y-%m-%d')
                    transaction_date = st.date_input(
                        "Transaction date",
                        value=default_date,
                        key="transaction_date"
                    )
                
                with col2:
                    submitted = st.form_submit_button(
                        "Save",
                        type="primary",
                        use_container_width=True,
                        on_click=lambda: setattr(st.session_state, 'form_submitted', True)
                    )

            if st.session_state.form_submitted:
                try:
                    if extracted_info['type'] in ['To Receive', 'To Pay']:
                        success = add_pending_transaction_to_sheet(
                            transaction_date.strftime('%Y-%m-%d'),  # Convert to string
                            extracted_info['amount'],
                            extracted_info['type'],
                            extracted_info['category'],
                            extracted_info.get('description', ''),
                            due_date.strftime('%Y-%m-%d')  # Convert to string
                        )
                    else:
                        success = add_transaction_to_sheet(
                            transaction_date.strftime('%Y-%m-%d'),  # Convert to string
                            extracted_info['amount'],
                            extracted_info['type'],
                            extracted_info['category'],
                            subcategory,
                            extracted_info.get('description', '')
                        )
                    
                    if success:
                        show_success_message(
                            transaction_date.strftime('%Y-%m-%d'),  # Convert to string
                            subcategory if 'subcategory' in locals() else None
                        )
                        st.session_state.current_transaction = None
                        st.session_state.form_submitted = False
                        st.rerun()
                    else:
                        st.error("Failed to save transaction. Please try again.")
                        st.session_state.form_submitted = False
                except Exception as e:
                    log.error(f"Failed to save transaction: {str(e)}")
                    st.error("An error occurred while saving the transaction. Please try again.")
                    st.session_state.form_submitted = False

def add_pending_transaction_to_sheet(date, amount, trans_type, category, description, due_date):
    try:
        # Verify sheets exist before adding transaction
        if not verify_sheets_setup():
            raise Exception("Failed to verify sheets setup")
            
        log.info(f"Starting pending transaction save: {date}, {amount}, {trans_type}, {category}, {description}, {due_date}")
        
        # Format the dates if they're datetime objects
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        if isinstance(due_date, datetime):
            due_date = due_date.strftime('%Y-%m-%d')
        
        # Ensure amount is a string
        amount = str(float(amount))
        
        # Prepare the values with initial status as 'Pending'
        values = [[str(date), amount, trans_type, category, description, str(due_date), 'Pending']]
        
        result = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range='Pending!A1:G1',
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body={'values': values}
        ).execute()
        
        log.info(f"✅ Pending transaction saved successfully: {result}")
        return True
        
    except Exception as e:
        log.error(f"❌ Failed to save pending transaction: {str(e)}")
        return False

def verify_sheets_setup():
    """Verify both Expenses and Pending sheets exist with correct headers"""
    try:
        # Get all sheets
        sheet_metadata = service.spreadsheets().get(spreadsheetId=SHEET_ID).execute()
        sheets = sheet_metadata.get('sheets', '')
        existing_sheets = {s.get("properties", {}).get("title") for s in sheets}
        
        # Check and initialize Expenses sheet
        if 'Expenses' not in existing_sheets:
            log.info("Creating new Expenses sheet...")
            body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': 'Expenses'
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=SHEET_ID,
                body=body
            ).execute()
            
            headers = [['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description']]
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range='Expenses!A1:F1',
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
        
        # Check and initialize Pending sheet
        if 'Pending' not in existing_sheets:
            log.info("Creating new Pending sheet...")
            body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': 'Pending'
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=SHEET_ID,
                body=body
            ).execute()
            
            headers = [['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status']]
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range='Pending!A1:G1',
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
            
        log.info("✨ Sheets verified and initialized")
        return True
    except Exception as e:
        log.error(f"❌ Failed to verify/initialize sheets: {str(e)}")
        return False

def main():
    """
    Main application function.
    Handles the core application flow and user interface.
    """
    try:
        log.info("🚀 Starting Finance Tracker application")
        
        # Initialize session state
        if 'sheets_verified' not in st.session_state:
            st.session_state.sheets_verified = False
        
        # Only verify sheets once
        if not st.session_state.sheets_verified:
            verify_sheets_setup()
            st.session_state.sheets_verified = True
        
        st.title("💰 Smart Finance Tracker")
        st.markdown(f"📊 [View Google Sheet]({get_sheet_url()})")
        st.divider()
        
        init_session_state()
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle chat input
        if prompt := st.chat_input("Tell me about your income or expense..."):
            log.debug(f"Received user input: {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process user input only if we don't have a current transaction
            if not st.session_state.current_transaction:
                extracted_info = process_user_input(prompt)
                st.session_state.current_transaction = extracted_info
                st.rerun()
            
        # Show transaction form if we have extracted info
        if st.session_state.current_transaction:
            show_transaction_form()
    
    except Exception as e:
        log.error(f"❌ Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()