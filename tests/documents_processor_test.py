import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAG.components.documents_processor import DocumentsProcessor, logger

NOTES_PATH = os.getenv("NOTES_PATH")

processor = DocumentsProcessor()

chunks = processor.process_documents(NOTES_PATH)

# 1. Базовый анализ chunks
print("=== BASIC CHUNKS ANALYSIS ===")
print(f"Total chunks: {len(chunks)}")
print(f"Type of chunks: {type(chunks)}")
print(f"Type of first element: {type(chunks[0]) if chunks else 'No chunks'}")

# 2. Анализ размеров chunks
print("\n=== CHUNKS SIZE ANALYSIS ===")
chunk_sizes = [len(chunk.page_content) for chunk in chunks]
print(f"Min chunk size: {min(chunk_sizes) if chunk_sizes else 0} chars")
print(f"Max chunk size: {max(chunk_sizes) if chunk_sizes else 0} chars")
print(f"Average chunk size: {sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0:.1f} chars")

# 3. Распределение размеров
print("\n=== SIZE DISTRIBUTION ===")
size_ranges = {
    "0-100": len([size for size in chunk_sizes if size <= 100]),
    "101-500": len([size for size in chunk_sizes if 101 <= size <= 500]),
    "501-1000": len([size for size in chunk_sizes if 501 <= size <= 1000]),
    "1001-1500": len([size for size in chunk_sizes if 1001 <= size <= 1500]),
    "1500+": len([size for size in chunk_sizes if size > 1500])
}

for range_name, count in size_ranges.items():
    percentage = (count / len(chunks)) * 100 if chunks else 0
    print(f"{range_name} chars: {count} chunks ({percentage:.1f}%)")

# 4. Анализ метаданных
print("\n=== METADATA ANALYSIS ===")
if chunks:
    first_chunk_metadata = chunks[0].metadata
    print("First chunk metadata keys:", list(first_chunk_metadata.keys()))
    print("First chunk metadata values:", first_chunk_metadata)

# 5. Анализ по исходным файлам
print("\n=== SOURCE FILE ANALYSIS ===")
from collections import defaultdict
file_chunks = defaultdict(list)

for chunk in chunks:
    source = chunk.metadata.get("source", "unknown")
    file_chunks[source].append(chunk)

print(f"Chunks distributed across {len(file_chunks)} files:")
for source, chunks_list in list(file_chunks.items())[:5]:  # Показываем первые 5 файлов
    print(f"  {os.path.basename(source)}: {len(chunks_list)} chunks")

# 6. Примеры chunks
print("\n=== SAMPLE CHUNKS ===")
for i, chunk in enumerate(chunks[:3]):  # Первые 3 chunks
    print(f"\n--- Chunk {i} ---")
    print(f"Size: {len(chunk.page_content)} chars")
    print(f"Chunk ID: {chunk.metadata.get('chunk_id')}")
    print(f"Source: {chunk.metadata.get('source')}")
    print(f"Content preview: {chunk.page_content[:100]}...")

# 7. Проверка целостности данных
print("\n=== DATA INTEGRITY CHECK ===")
empty_chunks = [chunk for chunk in chunks if not chunk.page_content.strip()]
print(f"Empty chunks: {len(empty_chunks)}")
print(f"Chunks with None content: {len([chunk for chunk in chunks if chunk.page_content is None])}")

# 8. Статистика по overlap (приблизительная)
print("\n=== OVERLAP ANALYSIS ===")
if len(chunks) > 1:
    # Проверяем overlap между первыми двумя chunks
    chunk1_content = chunks[0].page_content
    chunk2_content = chunks[1].page_content
    
    # Ищем overlap в конце первого и начале второго chunk
    overlap_size = 0
    for i in range(1, min(500, len(chunk1_content), len(chunk2_content))):
        if chunk1_content[-i:] == chunk2_content[:i]:
            overlap_size = i
            break
    
    print(f"Estimated overlap between first two chunks: {overlap_size} chars")


print("\n=== CHECK FULL DATA ===", end="\n")
for idx, chunk in enumerate(chunks[:5]):
    print(f"\n=== CHUNK №_{idx+1} START ===")
    print(chunk)
    print(f"\n=== CHUNK №_{idx+1} END ===")
