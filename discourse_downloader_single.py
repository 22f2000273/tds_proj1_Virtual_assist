import os
import json
import time
from datetime import datetime
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# === CONFIG ===
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34  # Verify in browser URL when viewing category
CHROME_PROFILE = os.path.expanduser("~/.config/google-chrome/Default")  # Auto-detects your profile
DATE_FROM = datetime(2025, 1, 1)
DATE_TO = datetime(2025, 4, 14)

def parse_date(date_str):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unparseable date: {date_str}")

def get_authenticated_context(playwright):
    """Ensures valid authentication state"""
    context = playwright.chromium.launch_persistent_context(
        CHROME_PROFILE,
        headless=False,
        channel="chrome",
        args=[
            "--disable-blink-features=AutomationControlled",
            "--start-maximized"
        ],
        ignore_https_errors=True,
        timeout=120000
    )
    
    # Verify login status
    page = context.new_page()
    page.goto(f"{BASE_URL}/latest")
    
    if "login" in page.url.lower():
        print("\n🔐 MANUAL LOGIN REQUIRED:")
        print("1. Click 'Continue with Google'")
        print("2. Complete 2FA if needed")
        print("3. Wait until FULLY logged in")
        print("4. Press ▶️ Resume in Playwright control bar\n")
        page.pause()
        
        # Final verification
        page.wait_for_selector('text="Your Dashboard"', timeout=30000)
    
    return context

# ... (keep all imports and config unchanged)

def scrape_posts(context):
    """Main scraping logic with proper rate limiting"""
    page = context.new_page()
    all_topics = []
    page_num = 0
    
    # Get all topics
    while True:
        print(f"📄 Fetching page {page_num}...")
        page.goto(f"{BASE_URL}/c/courses/tds-kb/{CATEGORY_ID}.json?page={page_num}", timeout=60000)
        time.sleep(2.5)
        
        try:
            data = json.loads(page.inner_text("pre"))
            topics = data.get("topic_list", {}).get("topics", [])
            if not topics: break
            all_topics.extend(topics)
            page_num += 1
        except:
            break

    # Process posts
    filtered_posts = []
    for topic in all_topics:
        created_at = parse_date(topic["created_at"])
        if not (DATE_FROM <= created_at <= DATE_TO):
            continue
            
        print(f"🔨 Processing topic {topic['id']}")
        page.goto(f"{BASE_URL}/t/{topic['slug']}/{topic['id']}.json", timeout=60000)
        time.sleep(1.5)
        
        try:
            topic_data = json.loads(page.inner_text("pre"))
            posts = topic_data.get("post_stream", {}).get("posts", [])
            accepted_answer_id = topic_data.get("accepted_answer_post_id")
            
            # Build reply count map
            reply_counter = {}
            for post in posts:
                reply_to = post.get("reply_to_post_number")
                if reply_to is not None:
                    reply_counter[reply_to] = reply_counter.get(reply_to, 0) + 1

            for post in posts:
                filtered_posts.append({
                    "topic_id": topic["id"],
                    "topic_title": topic.get("title", ""),
                    "category_id": topic.get("category_id"),
                    "tags": topic.get("tags", []),
                    "post_id": post["id"],
                    "post_number": post["post_number"],
                    "author": post.get("username", "unknown"),
                    "created_at": post["created_at"],
                    "updated_at": post.get("updated_at"),
                    "reply_to_post_number": post.get("reply_to_post_number"),
                    "is_reply": post.get("reply_to_post_number") is not None,
                    "reply_count": reply_counter.get(post["post_number"], 0),
                    "like_count": post.get("like_count", 0),
                    "is_accepted_answer": post["id"] == accepted_answer_id,
                    "mentioned_users": [u["username"] for u in post.get("mentioned_users", [])],
                    "url": f"{BASE_URL}/t/{topic['slug']}/{topic['id']}/{post['post_number']}",
                    "content": BeautifulSoup(post["cooked"], "html.parser").get_text().strip()
                })
        except:
            continue

    # Save output
    with open("discourse_posts.json", "w", encoding="utf-8") as f:
        json.dump(filtered_posts, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(filtered_posts)} posts")
    context.close()

# ... (keep rest of the code unchanged)


def main():
    with sync_playwright() as p:
        context = get_authenticated_context(p)
        scrape_posts(context)

if __name__ == "__main__":
    main()
