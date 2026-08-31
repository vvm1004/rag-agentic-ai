"""Lesson 02 — Advanced ChromaDB Metadata Filtering & Structured Search.

Demonstrates:
- Storing structured employee records with rich metadata.
- Generating semantic text documents from structured fields.
- Performing pure semantic search (Python developers, leadership roles).
- Executing exact metadata filters (by department, experience >= 10, California locations).
- Combining semantic search with complex metadata filters ($and, $gte, $in).
"""

from __future__ import annotations

import sys
from typing import Any, cast

# Safe UTF-8 reconfiguration for Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import CollectionMetadata, Where
from shared_functions import ef

# Creating an instance of ChromaClient
client: ClientAPI = chromadb.Client()

# Defining collection name
collection_name = "employee_collection"


def main() -> None:
    """Create employee collection and demonstrate advanced metadata filtering and search."""
    try:
        print("LESSON 02 — ADVANCED CHROMADB METADATA FILTERING")
        print("=" * 72)

        # Create or reset collection
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata=cast(
                CollectionMetadata,
                {
                    "description": "A collection for storing employee data",
                    "hnsw:space": "cosine",
                },
            ),
            embedding_function=ef,
        )

        print(f"Collection created: {collection.name}")

        # List of comprehensive employee records
        employees: list[dict[str, Any]] = [

            {
                "id": "employee_1",
                "name": "John Doe",
                "experience": 5,
                "department": "Engineering",
                "role": "Software Engineer",
                "skills": "Python, JavaScript, React, Node.js, databases",
                "location": "New York",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_2",
                "name": "Jane Smith",
                "experience": 8,
                "department": "Marketing",
                "role": "Marketing Manager",
                "skills": "Digital marketing, SEO, content strategy, analytics, social media",
                "location": "Los Angeles",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_3",
                "name": "Alice Johnson",
                "experience": 3,
                "department": "HR",
                "role": "HR Coordinator",
                "skills": "Recruitment, employee relations, HR policies, training programs",
                "location": "Chicago",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_4",
                "name": "Michael Brown",
                "experience": 12,
                "department": "Engineering",
                "role": "Senior Software Engineer",
                "skills": "Java, Spring Boot, microservices, cloud architecture, DevOps",
                "location": "San Francisco",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_5",
                "name": "Emily Wilson",
                "experience": 2,
                "department": "Marketing",
                "role": "Marketing Assistant",
                "skills": "Content creation, email marketing, market research, social media management",
                "location": "Austin",
                "employment_type": "Part-time",
            },
            {
                "id": "employee_6",
                "name": "David Lee",
                "experience": 15,
                "department": "Engineering",
                "role": "Engineering Manager",
                "skills": "Team leadership, project management, software architecture, mentoring",
                "location": "Seattle",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_7",
                "name": "Sarah Clark",
                "experience": 8,
                "department": "HR",
                "role": "HR Manager",
                "skills": "Performance management, compensation planning, policy development, conflict resolution",
                "location": "Boston",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_8",
                "name": "Chris Evans",
                "experience": 20,
                "department": "Engineering",
                "role": "Senior Architect",
                "skills": "System design, distributed systems, cloud platforms, technical strategy",
                "location": "New York",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_9",
                "name": "Jessica Taylor",
                "experience": 4,
                "department": "Marketing",
                "role": "Marketing Specialist",
                "skills": "Brand management, advertising campaigns, customer analytics, creative strategy",
                "location": "Miami",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_10",
                "name": "Alex Rodriguez",
                "experience": 18,
                "department": "Engineering",
                "role": "Lead Software Engineer",
                "skills": "Full-stack development, React, Python, machine learning, data science",
                "location": "Denver",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_11",
                "name": "Hannah White",
                "experience": 6,
                "department": "HR",
                "role": "HR Business Partner",
                "skills": "Strategic HR, organizational development, change management, employee engagement",
                "location": "Portland",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_12",
                "name": "Kevin Martinez",
                "experience": 10,
                "department": "Engineering",
                "role": "DevOps Engineer",
                "skills": "Docker, Kubernetes, AWS, CI/CD pipelines, infrastructure automation",
                "location": "Phoenix",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_13",
                "name": "Rachel Brown",
                "experience": 7,
                "department": "Marketing",
                "role": "Marketing Director",
                "skills": "Strategic marketing, team leadership, budget management, campaign optimization",
                "location": "Atlanta",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_14",
                "name": "Matthew Garcia",
                "experience": 3,
                "department": "Engineering",
                "role": "Junior Software Engineer",
                "skills": "JavaScript, HTML/CSS, basic backend development, learning frameworks",
                "location": "Dallas",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_15",
                "name": "Olivia Moore",
                "experience": 12,
                "department": "Engineering",
                "role": "Principal Engineer",
                "skills": "Technical leadership, system architecture, performance optimization, mentoring",
                "location": "San Francisco",
                "employment_type": "Full-time",
            },
        ]

        # Create comprehensive text documents for each employee
        employee_documents: list[str] = []
        for emp in employees:
            document = (
                f"{emp['role']} with {emp['experience']} years of experience in {emp['department']}. "
                f"Skills: {emp['skills']}. Located in {emp['location']}. "
                f"Employment type: {emp['employment_type']}."
            )
            employee_documents.append(document)

        # Add records to the Chroma collection
        collection.add(
            ids=[str(emp["id"]) for emp in employees],
            documents=employee_documents,
            metadatas=cast(
                Any,
                [
                    {
                        "name": str(emp["name"]),
                        "department": str(emp["department"]),
                        "role": str(emp["role"]),
                        "experience": int(emp["experience"]),
                        "location": str(emp["location"]),
                        "employment_type": str(emp["employment_type"]),
                    }
                    for emp in employees
                ],
            ),
        )


        all_items = collection.get()
        print(f"Collection populated with {len(all_items['documents'] or [])} employee profiles.\n")

        def perform_advanced_search(col: chromadb.Collection) -> None:
            print("=" * 60)
            print("PART A: SEMANTIC SIMILARITY SEARCH EXAMPLES")
            print("=" * 60)

            # 1. Search for Python developers
            print("\n1. Searching for Python developers:")
            query_1 = "Python developer with web development experience"
            results_1 = col.query(query_texts=[query_1], n_results=3)
            print(f"Query: '{query_1}'")
            ids_1 = results_1.get("ids")
            if ids_1 and len(ids_1) > 0 and len(ids_1[0]) > 0:
                docs_1 = results_1.get("documents")
                docs_list_1 = docs_1[0] if docs_1 is not None else [""] * len(ids_1[0])
                dists_1 = results_1.get("distances")
                dists_list_1 = dists_1[0] if dists_1 is not None else [0.0] * len(ids_1[0])
                metas_1 = results_1.get("metadatas")
                metas_list_1 = metas_1[0] if metas_1 is not None else [{}] * len(ids_1[0])

                for i in range(len(ids_1[0])):
                    doc_id = ids_1[0][i]
                    doc = docs_list_1[i]
                    dist = dists_list_1[i]
                    meta = metas_list_1[i] or {}
                    name = meta.get("name", doc_id)
                    role = meta.get("role", "")
                    dept = meta.get("department", "")
                    print(f"  {i+1}. {name} ({doc_id}) — Distance: {dist:.4f}")
                    print(f"     Role: {role}, Department: {dept}")
                    print(f"     Document snippet: {doc[:90]}...")

            # 2. Search for leadership roles
            print("\n2. Searching for leadership and management roles:")
            query_2 = "team leader manager with experience"
            results_2 = col.query(query_texts=[query_2], n_results=3)
            print(f"Query: '{query_2}'")
            ids_2 = results_2.get("ids")
            if ids_2 and len(ids_2) > 0 and len(ids_2[0]) > 0:
                dists_2 = results_2.get("distances")
                dists_list_2 = dists_2[0] if dists_2 is not None else [0.0] * len(ids_2[0])
                metas_2 = results_2.get("metadatas")
                metas_list_2 = metas_2[0] if metas_2 is not None else [{}] * len(ids_2[0])

                for i in range(len(ids_2[0])):
                    doc_id = ids_2[0][i]
                    dist = dists_list_2[i]
                    meta = metas_list_2[i] or {}
                    name = meta.get("name", doc_id)
                    role = meta.get("role", "")
                    exp = meta.get("experience", 0)
                    print(f"  {i+1}. {name} ({doc_id}) — Distance: {dist:.4f}")
                    print(f"     Role: {role}, Experience: {exp} years")

            print("\n" + "=" * 60)
            print("PART B: METADATA FILTERING EXAMPLES (EXACT CRITERIA)")
            print("=" * 60)

            # 3. Filter by department
            print("\n3. Finding all Engineering employees:")
            results_eng = col.get(where=cast(Where, {"department": "Engineering"}))
            ids_eng = results_eng.get("ids") or []
            metas_eng = results_eng.get("metadatas") or []
            print(f"Found {len(ids_eng)} Engineering employees:")
            for meta in metas_eng:
                if meta:
                    print(f"  - {meta.get('name')}: {meta.get('role')} ({meta.get('experience')} years)")

            # 4. Filter by experience range ($gte: 10)
            print("\n4. Finding employees with 10+ years of experience:")
            results_exp = col.get(where=cast(Where, {"experience": {"$gte": 10}}))
            ids_exp = results_exp.get("ids") or []
            metas_exp = results_exp.get("metadatas") or []
            print(f"Found {len(ids_exp)} senior employees:")
            for meta in metas_exp:
                if meta:
                    print(f"  - {meta.get('name')}: {meta.get('role')} ({meta.get('experience')} years in {meta.get('department')})")

            # 5. Filter by location ($in list)
            print("\n5. Finding employees in California (San Francisco, Los Angeles):")
            results_ca = col.get(where=cast(Where, {"location": {"$in": ["San Francisco", "Los Angeles"]}}))
            ids_ca = results_ca.get("ids") or []
            metas_ca = results_ca.get("metadatas") or []
            print(f"Found {len(ids_ca)} employees in California:")
            for meta in metas_ca:
                if meta:
                    print(f"  - {meta.get('name')}: {meta.get('role')} in {meta.get('location')}")

            print("\n" + "=" * 60)
            print("PART C: COMBINED SEARCH (SEMANTIC + COMPLEX METADATA FILTER)")
            print("=" * 60)

            # 6. Combined Search: Query + $and filter
            print("\n6. Finding senior Python developers in major tech cities:")
            query_comb = "senior Python developer full-stack"
            results_comb = col.query(
                query_texts=[query_comb],
                n_results=5,
                where=cast(
                    Where,
                    {
                        "$and": [
                            {"experience": {"$gte": 8}},
                            {"location": {"$in": ["San Francisco", "New York", "Seattle"]}},
                        ]
                    },
                ),
            )
            print(f"Query: '{query_comb}' with filters (>=8 yrs exp AND SF/NY/Seattle)")
            ids_comb = results_comb.get("ids")
            if ids_comb and len(ids_comb) > 0 and len(ids_comb[0]) > 0:
                docs_comb = results_comb.get("documents")
                docs_list_comb = docs_comb[0] if docs_comb is not None else [""] * len(ids_comb[0])
                dists_comb = results_comb.get("distances")
                dists_list_comb = dists_comb[0] if dists_comb is not None else [0.0] * len(ids_comb[0])
                metas_comb = results_comb.get("metadatas")
                metas_list_comb = metas_comb[0] if metas_comb is not None else [{}] * len(ids_comb[0])

                print(f"Found {len(ids_comb[0])} matching employees:")
                for i in range(len(ids_comb[0])):
                    doc_id = ids_comb[0][i]
                    doc = docs_list_comb[i]
                    dist = dists_list_comb[i]
                    meta = metas_list_comb[i] or {}
                    name = meta.get("name", doc_id)
                    role = meta.get("role", "")
                    loc = meta.get("location", "")
                    exp = meta.get("experience", 0)
                    print(f"  {i+1}. {name} ({doc_id}) — Distance: {dist:.4f}")
                    print(f"     {role} in {loc} ({exp} years)")
                    print(f"     Document snippet: {doc[:80]}...")
            else:
                print(f'No matching documents found for "{query_comb}" with given filters.')

        perform_advanced_search(collection)

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
