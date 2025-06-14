#!/usr/bin/env python3
"""
Database diagnosis and fix script for the RAG application
This script will help identify and fix database issues
"""

import os
import sqlite3
import json
from pathlib import Path

def diagnose_database():
    """Diagnose the current database state"""
    db_path = "knowledge_base.db"
    
    print("=== Database Diagnosis ===")
    print(f"Looking for database at: {os.path.abspath(db_path)}")
    
    # Check if file exists
    if not os.path.exists(db_path):
        print("❌ Database file does not exist")
        return False
    
    print(f"✅ Database file exists")
    print(f"File size: {os.path.getsize(db_path)} bytes")
    
    # Check if it's a valid SQLite database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Try to execute a simple query
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"✅ Valid SQLite database")
        print(f"Tables found: {[table[0] for table in tables]}")
        
        # Check for required tables
        required_tables = ['discourse_chunks', 'markdown_chunks']
        existing_tables = [table[0] for table in tables]
        
        for table in required_tables:
            if table in existing_tables:
                print(f"✅ Table '{table}' exists")
                
                # Count rows
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   - Row count: {count}")
                
                # Check for embeddings
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL")
                embedding_count = cursor.fetchone()[0]
                print(f"   - Rows with embeddings: {embedding_count}")
                
            else:
                print(f"❌ Table '{table}' missing")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_fresh_database():
    """Create a fresh database with the required schema"""
    db_path = "knowledge_base.db"
    
    print("\n=== Creating Fresh Database ===")
    
    # Remove existing file if it exists
    if os.path.exists(db_path):
        print("Removing existing database file...")
        os.remove(db_path)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create discourse_chunks table
        print("Creating discourse_chunks table...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS discourse_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            topic_id INTEGER,
            topic_title TEXT,
            post_number INTEGER,
            author TEXT,
            created_at TEXT,
            likes INTEGER,
            chunk_index INTEGER,
            content TEXT,
            url TEXT,
            embedding BLOB
        )
        ''')
        
        # Create markdown_chunks table
        print("Creating markdown_chunks table...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_title TEXT,
            original_url TEXT,
            downloaded_at TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB
        )
        ''')
        
        # Create indexes for better performance
        print("Creating indexes...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discourse_post_id ON discourse_chunks(post_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discourse_topic_id ON discourse_chunks(topic_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_markdown_title ON markdown_chunks(doc_title)')
        
        conn.commit()
        conn.close()
        
        print("✅ Fresh database created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

def add_sample_data():
    """Add some sample data for testing"""
    db_path = "knowledge_base.db"
    
    print("\n=== Adding Sample Data ===")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Add sample discourse chunk
        sample_embedding = json.dumps([0.1] * 1536)  # Dummy embedding
        
        cursor.execute('''
        INSERT INTO discourse_chunks 
        (post_id, topic_id, topic_title, post_number, author, created_at, likes, chunk_index, content, url, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            1, 1, "Sample Topic", 1, "test_user", "2024-01-01", 0, 0,
            "This is a sample discourse post content for testing the RAG system.",
            "https://discourse.example.com/t/sample-topic/1",
            sample_embedding
        ))
        
        # Add sample markdown chunk
        cursor.execute('''
        INSERT INTO markdown_chunks 
        (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            "Sample Documentation",
            "https://docs.example.com/sample",
            "2024-01-01",
            0,
            "This is sample documentation content for testing the RAG system.",
            sample_embedding
        ))
        
        conn.commit()
        conn.close()
        
        print("✅ Sample data added successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error adding sample data: {e}")
        return False

def main():
    """Main function to run the diagnosis and fix"""
    print("RAG Database Diagnosis and Fix Tool")
    print("=" * 40)
    
    # First, diagnose the current state
    is_healthy = diagnose_database()
    
    if not is_healthy:
        print("\n🔧 Database issues detected. Creating fresh database...")
        if create_fresh_database():
            print("\n🎯 Adding sample data for testing...")
            add_sample_data()
            
            print("\n✅ Database setup complete!")
            print("\nNext steps:")
            print("1. Run your application to test the connection")
            print("2. Add your actual data using your data ingestion scripts")
            print("3. Generate embeddings for your content")
        else:
            print("\n❌ Failed to create database. Check file permissions.")
    else:
        print("\n✅ Database appears to be healthy!")

if __name__ == "__main__":
    main()