from langchain_community.document_loaders import PyPDFLoader
import os

def test_pypdfloader():
    """
    Simple test to check if PyPDFLoader works with HSC PDF
    """
    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    
    print("🧪 Testing PyPDFLoader")
    print("=" * 30)
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        print("🔄 Loading PDF...")
        loader = PyPDFLoader(pdf_path)
        
        print("🔄 Extracting pages...")
        pages = loader.load()
        
        print(f"✅ Successfully loaded {len(pages)} pages")
        
        # Test first few pages
        for i in range(min(3, len(pages))):
            page = pages[i]
            content = page.page_content
            print(f"\n📄 Page {i+1}:")
            print(f"   Characters: {len(content)}")
            print(f"   Preview: {content[:150]}...")
            
            # Save sample to file
            with open(f"sample_page_{i+1}.txt", "w", encoding="utf-8") as f:
                f.write(content)
        
        print(f"\n✅ PyPDFLoader test completed!")
        print(f"📁 Sample files saved: sample_page_1.txt, sample_page_2.txt, sample_page_3.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_pypdfloader()
