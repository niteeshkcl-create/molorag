import sys 
import os
sys.path.append('../')
from utils import prepare_files, get_cur_time
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import argparse 

def index_single_pdf(filepath, doc_id, embeddings, default_parser=True):
    if default_parser:
        loader = PyPDFLoader(filepath)
    else:
        loader = UnstructuredFileLoader(filepath)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, 
                                                   chunk_overlap=args.overlap, 
                                                   add_start_index=True)
    chunks = text_splitter.split_documents(pages)
    
    cur_savepath = f"{save_dir}/{doc_id}"
    faiss_index = FAISS.from_documents(chunks, embeddings)
    faiss_index.save_local(cur_savepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong")
    parser.add_argument("--save_dir", type=str, default="../tmp/tmp_dbs")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    
    save_dir = f"{args.save_dir}/{args.dataset}" 
    os.makedirs(save_dir, exist_ok=True)

    print(f"{get_cur_time()} - Loading local embeddings: {args.model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=args.model_name)

    pdf_files = prepare_files(root_dir=f"../dataset/{args.dataset}", suffix=".pdf")
    # Limit to 5 for fast testing as requested
    pdf_files = pdf_files[:5]
    
    for cur_pdf in tqdm(pdf_files, desc=f"Indexing (Subset) in {args.dataset}"):
        doc_id = cur_pdf.replace('.pdf', '')
        if os.path.exists(f"{save_dir}/{doc_id}"):
            continue 
        
        try: 
            index_single_pdf(filepath=f"../dataset/{args.dataset}/{cur_pdf}", doc_id=doc_id, embeddings=embeddings)
        except Exception as e:
            print(f"[ERROR] processing {cur_pdf}: {e}")

    print(f"{get_cur_time()} - Finished Indexing Subset! ")
