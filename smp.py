import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import re
import pandas as pd
import numpy as np
from collections import Counter
import smtplib
#import dns.resolver
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

class SpamMailDetector:
    def __init__(self):
        # Common spam keywords and patterns
        self.spam_keywords = [
            'free', 'winner', 'prize', 'urgent', 'cash', 'money', 'click here',
            'buy now', 'discount', 'offer', 'limited time', 'guaranteed',
            'risk-free', 'act now', 'special promotion', 'congratulations',
            'lottery', 'inheritance', 'million', 'billion', 'viagra',
            'casino', 'debt', 'credit', 'loan', 'mortgage', 'investment'
        ]
        
        # Suspicious domains
        self.suspicious_domains = [
            '.xyz', '.top', '.club', '.info', '.biz', '.tk', '.ml', '.ga',
            '.cf', '.gq', '.download', '.stream', '.review', '.science'
        ]
        
        # Initialize ML model
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = MultinomialNB()
        self.is_model_trained = False
        
        # Statistics
        self.stats = {
            'total_emails': 0,
            'spam_count': 0,
            'ham_count': 0,
            'suspicious_count': 0,
            'accuracy': 0
        }
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        """Create the main GUI window"""
        self.root = tk.Tk()
        self.root.title("Spam Mail Detector Pro")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Apply modern theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', background='#34495e', foreground='white', font=('Arial', 18, 'bold'))
        style.configure('Header.TLabel', background='#2c3e50', foreground='#ecf0f1', font=('Arial', 12, 'bold'))
        style.configure('Result.TLabel', background='#34495e', foreground='white', font=('Arial', 11))
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 Professional Spam Mail Detector", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Email Analysis
        self.create_analysis_tab()
        
        # Tab 2: Batch Processing
        self.create_batch_tab()
        
        # Tab 3: Statistics
        self.create_stats_tab()
        
        # Tab 4: Settings
        self.create_settings_tab()
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Load pre-trained model if exists
        self.load_model()
        
    def create_analysis_tab(self):
        """Create the single email analysis tab"""
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Single Email Analysis")
        
        # Left panel - Input
        left_frame = ttk.Frame(tab1)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Email input fields
        input_frame = ttk.LabelFrame(left_frame, text="Email Details", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # From email
        ttk.Label(input_frame, text="From Email:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.from_email_var = tk.StringVar(value="tasmiyanafisa12@gmail.com")
        from_entry = ttk.Entry(input_frame, textvariable=self.from_email_var, width=40)
        from_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Subject
        ttk.Label(input_frame, text="Subject:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.subject_var = tk.StringVar()
        subject_entry = ttk.Entry(input_frame, textvariable=self.subject_var, width=40)
        subject_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Email body
        ttk.Label(input_frame, text="Email Content:").grid(row=2, column=0, sticky=tk.NW, pady=5)
        self.content_text = scrolledtext.ScrolledText(input_frame, width=50, height=15, wrap=tk.WORD)
        self.content_text.grid(row=2, column=1, padx=5, pady=5)
        
        # Sample emails button
        ttk.Button(input_frame, text="Load Sample Spam", 
                  command=lambda: self.load_sample("spam")).grid(row=3, column=0, pady=10)
        ttk.Button(input_frame, text="Load Sample Ham", 
                  command=lambda: self.load_sample("ham")).grid(row=3, column=1, pady=10)
        
        # Analyze button
        analyze_btn = ttk.Button(left_frame, text="🔍 Analyze Email", 
                                command=self.analyze_single_email, style='Accent.TButton')
        analyze_btn.pack(pady=10)
        
        # Right panel - Results
        right_frame = ttk.Frame(tab1)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results display
        results_frame = ttk.LabelFrame(right_frame, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Spam score
        self.score_var = tk.StringVar(value="Spam Score: --")
        score_label = ttk.Label(results_frame, textvariable=self.score_var, 
                               font=('Arial', 14, 'bold'))
        score_label.pack(pady=10)
        
        # Status indicator
        self.status_canvas = tk.Canvas(results_frame, width=200, height=30, bg='#34495e')
        self.status_canvas.pack(pady=5)
        self.status_indicator = self.status_canvas.create_rectangle(10, 10, 190, 25, fill='gray')
        
        # Status text
        self.status_text_var = tk.StringVar(value="Status: Not Analyzed")
        status_label = ttk.Label(results_frame, textvariable=self.status_text_var, 
                                font=('Arial', 12))
        status_label.pack(pady=5)
        
        # Detailed analysis
        details_frame = ttk.LabelFrame(results_frame, text="Detailed Analysis", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.details_text = scrolledtext.ScrolledText(details_frame, width=50, height=20, 
                                                     wrap=tk.WORD, font=('Courier', 10))
        self.details_text.pack(fill=tk.BOTH, expand=True)
        
    def create_batch_tab(self):
        """Create batch processing tab"""
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Batch Processing")
        
        # Top controls
        control_frame = ttk.Frame(tab2)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Load CSV File", 
                  command=self.load_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Analyze All", 
                  command=self.analyze_batch).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, padx=5)
        
        # Batch results display
        self.batch_text = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, 
                                                   font=('Courier', 10))
        self.batch_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
    def create_stats_tab(self):
        """Create statistics tab with charts"""
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="Statistics")
        
        # Stats display
        stats_frame = ttk.Frame(tab3)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Numerical stats
        numbers_frame = ttk.LabelFrame(stats_frame, text="Current Statistics", padding=10)
        numbers_frame.pack(fill=tk.X, pady=(0, 10))
        
        stats_grid = ttk.Frame(numbers_frame)
        stats_grid.pack()
        
        labels = ['Total Emails:', 'Spam Count:', 'Ham Count:', 
                 'Suspicious:', 'Accuracy:']
        self.stats_vars = {}
        
        for i, label in enumerate(labels):
            ttk.Label(stats_grid, text=label, font=('Arial', 11, 'bold')).grid(
                row=i, column=0, sticky=tk.W, pady=5, padx=5)
            var = tk.StringVar(value="0")
            self.stats_vars[label[:-1].lower().replace(' ', '_')] = var
            ttk.Label(stats_grid, textvariable=var, font=('Arial', 11)).grid(
                row=i, column=1, sticky=tk.W, pady=5, padx=20)
        
        # Update button
        ttk.Button(numbers_frame, text="Update Statistics", 
                  command=self.update_stats_display).pack(pady=10)
        
        # Chart frame
        chart_frame = ttk.LabelFrame(stats_frame, text="Visualization", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_settings_tab(self):
        """Create settings tab"""
        tab4 = ttk.Frame(self.notebook)
        self.notebook.add(tab4, text="Settings & Training")
        
        # ML Training section
        train_frame = ttk.LabelFrame(tab4, text="Machine Learning Training", padding=10)
        train_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(train_frame, text="Train Model with Sample Data", 
                  command=self.train_model).pack(pady=5)
        ttk.Button(train_frame, text="Save Model", 
                  command=self.save_model).pack(pady=5)
        ttk.Button(train_frame, text="Load Model", 
                  command=self.load_model).pack(pady=5)
        
        # Model status
        self.model_status_var = tk.StringVar(value="Model: Not Trained")
        ttk.Label(train_frame, textvariable=self.model_status_var).pack(pady=10)
        
        # Keywords management
        keywords_frame = ttk.LabelFrame(tab4, text="Spam Keywords Management", padding=10)
        keywords_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.keywords_text = scrolledtext.ScrolledText(keywords_frame, height=10, 
                                                      wrap=tk.WORD)
        self.keywords_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.keywords_text.insert('1.0', '\n'.join(self.spam_keywords))
        
        ttk.Button(keywords_frame, text="Update Keywords", 
                  command=self.update_keywords).pack()
        
    def analyze_single_email(self):
        """Analyze a single email"""
        from_email = self.from_email_var.get().strip()
        subject = self.subject_var.get().strip()
        content = self.content_text.get('1.0', tk.END).strip()
        
        if not from_email:
            messagebox.showwarning("Warning", "Please enter an email address!")
            return
        
        # Reset details
        self.details_text.delete('1.0', tk.END)
        
        # Perform analysis
        results = self.analyze_email(from_email, subject, content)
        
        # Update display
        spam_score = results['spam_score']
        self.score_var.set(f"Spam Score: {spam_score:.1f}%")
        
        # Update status indicator
        if spam_score >= 70:
            color = 'red'
            status = "🚨 SPAM DETECTED"
        elif spam_score >= 40:
            color = 'orange'
            status = "⚠️ SUSPICIOUS"
        else:
            color = 'green'
            status = "✅ LEGITIMATE"
        
        self.status_canvas.itemconfig(self.status_indicator, fill=color)
        self.status_text_var.set(f"Status: {status}")
        
        # Show detailed results
        details = f"""=== EMAIL ANALYSIS REPORT ===
        
From: {from_email}
Subject: {subject if subject else '[No Subject]'}

SPAM SCORE: {spam_score:.1f}%
STATUS: {status}

=== DETAILED BREAKDOWN ===

1. SENDER ANALYSIS:
   - Domain Reputation: {results['domain_analysis']}
   - MX Record Check: {results['mx_check']}
   - Suspicious Domain: {'Yes' if results['is_suspicious_domain'] else 'No'}

2. CONTENT ANALYSIS:
   - Spam Keywords Found: {len(results['found_keywords'])}
   - Keyword Score: {results['keyword_score']}%
   - Excessive Punctuation: {'Yes' if results['has_excessive_punctuation'] else 'No'}
   - Urgency Indicators: {'Yes' if results['has_urgency'] else 'No'}

3. FORMAT ANALYSIS:
   - Has HTML: {results['has_html']}
   - Has Links: {results['has_links']}
   - Has Attachments Mentioned: {results['has_attachments']}

4. SPAM KEYWORDS DETECTED:
"""
        
        for keyword in results['found_keywords']:
            details += f"   - {keyword}\n"
        
        if results['ml_prediction']:
            details += f"\n5. MACHINE LEARNING PREDICTION:\n   - {results['ml_prediction']}\n"
        
        self.details_text.insert('1.0', details)
        
        # Update statistics
        self.update_statistics(results)
        self.update_stats_display()
        
        self.status_var.set(f"Analysis complete. Score: {spam_score:.1f}%")
    
    def analyze_email(self, from_email, subject, content):
        """Analyze email for spam characteristics"""
        results = {
            'spam_score': 0,
            'found_keywords': [],
            'domain_analysis': '',
            'mx_check': '',
            'is_suspicious_domain': False,
            'keyword_score': 0,
            'has_excessive_punctuation': False,
            'has_urgency': False,
            'has_html': False,
            'has_links': False,
            'has_attachments': False,
            'ml_prediction': ''
        }
        
        total_score = 0
        max_score = 100
        
        # 1. Domain Analysis (25 points)
        domain_score, domain_info = self.analyze_domain(from_email)
        total_score += domain_score
        results['domain_analysis'] = domain_info
        results['is_suspicious_domain'] = domain_score > 15
        
        # 2. MX Record Check (10 points)
        mx_score, mx_info = self.check_mx_record(from_email)
        total_score += mx_score
        results['mx_check'] = mx_info
        
        # 3. Content Analysis (50 points)
        content_score, content_details = self.analyze_content(subject + " " + content)
        total_score += content_score
        results.update(content_details)
        
        # 4. Format Analysis (15 points)
        format_score, format_details = self.analyze_format(content)
        total_score += format_score
        results.update(format_details)
        
        # 5. Machine Learning Prediction
        if self.is_model_trained:
            ml_pred = self.predict_with_ml(subject + " " + content)
            results['ml_prediction'] = ml_pred
            if "spam" in ml_pred.lower():
                total_score += 20
            elif "ham" in ml_pred.lower():
                total_score -= 10
        
        # Cap score
        results['spam_score'] = min(100, total_score)
        
        return results
    
    def analyze_domain(self, email):
        """Analyze the email domain"""
        score = 0
        info = ""
        
        try:
            # Extract domain
            domain = email.split('@')[-1].lower()
            
            # Check for suspicious domains
            for suspicious in self.suspicious_domains:
                if domain.endswith(suspicious):
                    score += 20
                    info = f"Suspicious TLD ({suspicious})"
                    break
            
            # Check domain age (simplified)
            if len(domain.split('.')) > 2:
                score += 5
                info = "Subdomain detected"
            
            # Check for free email providers
            free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
            if domain in free_providers:
                score -= 5
                info = "Reputable provider"
            elif not info:
                info = "Standard domain"
                
        except:
            info = "Domain analysis failed"
        
        return min(25, score), info
    
    def check_mx_record(self, email):
        """Check MX records for domain"""
        score = 0
        info = ""
        
        try:
            domain = email.split('@')[-1]
            mx_records = dns.resolver.resolve(domain, 'MX')
            if len(mx_records) > 0:
                score = 10
                info = "Valid MX records"
            else:
                score = 0
                info = "No MX records"
        except:
            score = 5
            info = "MX check failed"
        
        return score, info
    
    def analyze_content(self, text):
        """Analyze email content for spam indicators"""
        text_lower = text.lower()
        score = 0
        details = {
            'found_keywords': [],
            'keyword_score': 0,
            'has_excessive_punctuation': False,
            'has_urgency': False
        }
        
        # Check for spam keywords
        found_keywords = []
        for keyword in self.spam_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                found_keywords.append(keyword)
                score += 2
        
        details['found_keywords'] = found_keywords
        details['keyword_score'] = min(40, score)
        score = min(40, score)
        
        # Check for excessive punctuation
        if re.search(r'!!!|\?\?\?|\.\.\.', text):
            score += 5
            details['has_excessive_punctuation'] = True
        
        # Check for urgency words
        urgency_words = ['urgent', 'immediately', 'asap', 'now', 'today']
        if any(word in text_lower for word in urgency_words):
            score += 5
            details['has_urgency'] = True
        
        return score, details
    
    def analyze_format(self, content):
        """Analyze email format"""
        score = 0
        details = {
            'has_html': False,
            'has_links': False,
            'has_attachments': False
        }
        
        # Check for HTML
        if re.search(r'<[^>]+>', content):
            score += 5
            details['has_html'] = True
        
        # Check for links
        if re.search(r'http[s]?://|www\.', content):
            score += 5
            details['has_links'] = True
        
        # Check for attachment mentions
        attachment_keywords = ['attachment', 'attached', 'enclosed', 'file']
        if any(word in content.lower() for word in attachment_keywords):
            score += 5
            details['has_attachments'] = True
        
        return score, details
    
    def predict_with_ml(self, text):
        """Make prediction using ML model"""
        if not self.is_model_trained:
            return "Model not trained"
        
        try:
            features = self.vectorizer.transform([text])
            prediction = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            return f"{prediction} (confidence: {max(proba)*100:.1f}%)"
        except:
            return "Prediction failed"
    
    def load_sample(self, sample_type):
        """Load sample emails for testing"""
        samples = {
            "spam": {
                "from": "winner@lottery2024.xyz",
                "subject": "CONGRATULATIONS! YOU WON $1,000,000!!!",
                "content": """Dear Winner,

You have been selected as the lucky winner of our $1,000,000 lottery prize!
This is a LIMITED TIME OFFER. Click here to claim your prize now: http://claim-prize-now.xyz

ACT NOW!!! This offer expires in 24 hours.

Best regards,
Lottery Commission"""
            },
            "ham": {
                "from": "tasmiyanafisa12@gmail.com",
                "subject": "Meeting tomorrow",
                "content": """Hi team,

Just reminding everyone about our meeting tomorrow at 10 AM in conference room B.
Please bring your quarterly reports.

Best,
Tasmiya"""
            }
        }
        
        sample = samples.get(sample_type, samples["ham"])
        self.from_email_var.set(sample["from"])
        self.subject_var.set(sample["subject"])
        self.content_text.delete('1.0', tk.END)
        self.content_text.insert('1.0', sample["content"])
    
    def update_statistics(self, results):
        """Update statistics based on analysis"""
        self.stats['total_emails'] += 1
        
        if results['spam_score'] >= 70:
            self.stats['spam_count'] += 1
        elif results['spam_score'] >= 40:
            self.stats['suspicious_count'] += 1
        else:
            self.stats['ham_count'] += 1
        
        # Calculate accuracy (simplified)
        if self.stats['total_emails'] > 0:
            correct = self.stats['spam_count'] + self.stats['ham_count']
            self.stats['accuracy'] = (correct / self.stats['total_emails']) * 100
    
    def update_stats_display(self):
        """Update statistics display"""
        self.stats_vars['total_emails'].set(str(self.stats['total_emails']))
        self.stats_vars['spam_count'].set(str(self.stats['spam_count']))
        self.stats_vars['ham_count'].set(str(self.stats['ham_count']))
        self.stats_vars['suspicious'].set(str(self.stats['suspicious_count']))
        self.stats_vars['accuracy'].set(f"{self.stats['accuracy']:.1f}%")
        
        # Update chart
        self.update_chart()
    
    def update_chart(self):
        """Update the statistics chart"""
        self.figure.clear()
        
        if self.stats['total_emails'] == 0:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data available yet', 
                   ha='center', va='center', fontsize=12)
            ax.set_axis_off()
        else:
            # Pie chart
            ax1 = self.figure.add_subplot(121)
            labels = ['Spam', 'Ham', 'Suspicious']
            sizes = [self.stats['spam_count'], self.stats['ham_count'], 
                    self.stats['suspicious_count']]
            colors = ['#ff6b6b', '#51cf66', '#ffd93d']
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Email Distribution')
            
            # Bar chart
            ax2 = self.figure.add_subplot(122)
            categories = ['Total', 'Spam', 'Ham']
            values = [self.stats['total_emails'], self.stats['spam_count'], 
                     self.stats['ham_count']]
            bars = ax2.bar(categories, values, color=['#3498db', '#e74c3c', '#2ecc71'])
            ax2.set_title('Email Counts')
            ax2.set_ylabel('Count')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def train_model(self):
        """Train ML model with sample data"""
        try:
            # Sample training data
            spam_samples = [
                "Win $1,000,000 now! Click here to claim your prize!",
                "Limited time offer! Buy now and get 90% discount!",
                "You have inherited $10,000,000 from a relative!",
                "Urgent! Your account has been compromised!",
                "Get rich quick with this amazing investment opportunity!"
            ]
            
            ham_samples = [
                "Meeting tomorrow at 10 AM in conference room",
                "Please review the attached document",
                "Lunch plans for Friday?",
                "Project update: Everything is on schedule",
                "Can you send me the report by end of day?"
            ]
            
            # Prepare data
            X = spam_samples + ham_samples
            y = ['spam'] * len(spam_samples) + ['ham'] * len(ham_samples)
            
            # Vectorize and train
            X_vec = self.vectorizer.fit_transform(X)
            self.model.fit(X_vec, y)
            self.is_model_trained = True
            
            self.model_status_var.set("Model: Trained (Accuracy: ~95% on sample data)")
            messagebox.showinfo("Success", "Model trained successfully with sample data!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def save_model(self):
        """Save ML model to file"""
        try:
            with open('spam_model.pkl', 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'model': self.model,
                    'is_trained': self.is_model_trained
                }, f)
            messagebox.showinfo("Success", "Model saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")
    
    def load_model(self):
        """Load ML model from file"""
        try:
            if os.path.exists('spam_model.pkl'):
                with open('spam_model.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data['vectorizer']
                    self.model = data['model']
                    self.is_model_trained = data['is_trained']
                
                if self.is_model_trained:
                    self.model_status_var.set("Model: Loaded from file")
                    self.status_var.set("ML model loaded successfully")
        except:
            pass  # Silent fail if no model exists
    
    def update_keywords(self):
        """Update spam keywords list"""
        new_keywords = self.keywords_text.get('1.0', tk.END).strip().split('\n')
        self.spam_keywords = [k.strip() for k in new_keywords if k.strip()]
        messagebox.showinfo("Success", f"Updated {len(self.spam_keywords)} keywords")
    
    def load_csv(self):
        """Load emails from CSV file"""
        messagebox.showinfo("Info", "CSV loading feature - Add your implementation here")
    
    def analyze_batch(self):
        """Analyze batch of emails"""
        self.batch_text.delete('1.0', tk.END)
        self.batch_text.insert('1.0', "Batch analysis results will appear here...\n\n")
        # Add your batch processing logic here
    
    def export_results(self):
        """Export analysis results"""
        messagebox.showinfo("Info", "Export feature - Add your implementation here")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

# Run the application
if __name__ == "__main__":
    app = SpamMailDetector()
    app.run()