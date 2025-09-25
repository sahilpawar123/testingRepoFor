# 💰 Smart Finance Tracker

A modern, AI-powered expense tracking application built with Streamlit that helps you manage your personal finances through natural language input and intelligent categorization.

## ✨ Features

### Core Features
- 🤖 Natural Language Processing for expense/income entry
  - Intelligent parsing of casual inputs like "Spent $50 on groceries yesterday"
  - Automatic date detection from natural language
  - Smart categorization of transactions
- 💬 Chat-like interface for easy data entry
- 📈 Google Sheets integration for reliable data storage
- 📱 Responsive design for both desktop and mobile

### Transaction Management
- 💸 Support for multiple transaction types:
  - Regular income and expenses
  - Pending payments (To Pay)
  - Pending receivables (To Receive)
- 🏷️ Hierarchical categorization with categories and subcategories
- 📝 Detailed transaction descriptions
- 📅 Flexible date handling for both transaction and due dates

### Analytics & Insights
- 📊 Comprehensive financial analytics:
  - Overview dashboard with key metrics
  - Income analytics
  - Expense analytics
  - Pending transactions summary
- 📈 Advanced visualizations:
  - Monthly income vs expenses trends
  - Category-wise breakdowns
  - Top income sources and expense categories
  - Weekly spending patterns
- 💡 Smart insights:
  - Weekday vs weekend spending analysis
  - Fixed vs variable expense detection
  - Week-of-month spending patterns
- 📅 Flexible date filtering:
  - All time view
  - Yearly analysis
  - Monthly analysis
  - Custom date ranges

## 🚀 Getting Started

### Prerequisites

1. **Python Environment:**
   - Python 3.8 or higher
   - pip package manager

2. **Google Cloud Setup:**
   - Google account
   - Google Cloud project
   - Google Sheets API enabled
   - Service account with appropriate permissions

3. **Gemini AI API:**
   - Gemini AI API key (for natural language processing)

### Detailed Setup Process

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Spritan/expense_tracker
   cd expense_tracker
   ```

2. **Install Dependencies:**
   ```bash
   pip install uv
   uv sync
   ```

3. **Google Cloud Platform Setup:**
   a. Create a new project in Google Cloud Console
   b. Enable the Google Sheets API:
      - Go to APIs & Services > Library
      - Search for "Google Sheets API"
      - Click Enable
   c. Create a service account:
      - Go to APIs & Services > Credentials
      - Click "Create Credentials" > "Service Account"
      - Fill in the service account details
      - Create and download the JSON key file
   d. Set up Google Sheet:
      - Create a new Google Sheet
      - Share it with the service account email
      - Note down the Sheet ID from the URL

4. **Environment Configuration:**
   Create a `.env` file in the project root with:
   ```
   GOOGLE_SHEETS_CREDENTIALS=path/to/your/credentials.json
   GOOGLE_SHEET_ID=your_google_sheet_id
   GEMINI_API_KEY=your_gemini_api_key
   ```

5. **Initialize Google Sheet:**
   The application will automatically set up the required columns:
   - Date
   - Amount
   - Type (Income/Expense/To Pay/To Receive)
   - Category
   - Subcategory
   - Description
   - Due Date (for pending transactions)

### Running the Application

1. **Start the Application:**
   ```bash
   uv run streamlit run Home.py
   ```

2. **First-Time Setup:**
   - The application will automatically verify and initialize the Google Sheet structure
   - You'll see a success message if everything is configured correctly

3. **Verify Installation:**
   - Check if the chat interface appears
   - Try adding a sample transaction
   - Verify if the analytics tabs are working

## 🔧 Configuration

### Transaction Categories

**Income Categories:**
- Salary
- Investments
- Business
- Other Income

**Expense Categories:**
- Food & Dining
- Shopping
- Transportation
- Bills & Utilities
- Entertainment
- Health & Wellness
- Other Expenses

### Pending Transaction Types
- To Pay (for upcoming payments)
- To Receive (for expected income)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by Google's Gemini AI
- Uses Google Sheets API for data storage
- Visualizations powered by Plotly

## 💡 Support

For support:
1. Check the documentation above
2. Open an issue in the GitHub repository
3. Contact the maintainers

## 🔒 Security Note

- Never commit your `.env` file or credentials to version control
- Keep your API keys and credentials secure
- Regularly rotate your service account keys
- Follow the principle of least privilege when setting up service accounts
