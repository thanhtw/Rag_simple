"""
Configuration manager for the RAG-based code review application.

This module handles application settings, UI rendering, and integration
with LLM and data processing components.
"""

import streamlit as st
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer
import chromadb
from constants import (
    EN, VI, NONE, USER, ASSISTANT, ENGLISH, VIETNAMESE,
    LOCAL_LLM, DEFAULT_LOCAL_LLM, DB, VECTOR_SEARCH, HYDE_SEARCH,
    OLLAMA_MODEL_OPTIONS, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_NUM_DOCS_RETRIEVAL
)
from data_processor import DataProcessor
from Ollama import LocalLlms, OllamaManager


class ConfigManager:
    """
    Manages configuration settings and UI components for the application.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.llm_options = {
            "Online": "Online",
            "Local (Ollama)": "Local (Ollama)"
        }
        self.initialize_state()
        self.data_processor = DataProcessor(self)
        self.ollama_manager = OllamaManager()
        self.ollama_endpoint = os.getenv('OLLAMA_ENDPOINT') or "http://localhost:11434"

    def initialize_state(self):
        """Initialize all session state variables with default values."""
        defaults = {
            "language": None,
            "embedding_model": None,
            "embedding_model_name": None,
            "llm_type": LOCAL_LLM,
            "llm_name": DEFAULT_LOCAL_LLM,
            "llm_model": None,
            "local_llms": None,
            "client": chromadb.PersistentClient("db"),
            "active_collections": {},
            "search_option": VECTOR_SEARCH,
            "open_dialog": None,
            "source_data": "UPLOAD",
            "chunks_df": pd.DataFrame(),
            "random_collection_name": None,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "number_docs_retrieval": DEFAULT_NUM_DOCS_RETRIEVAL,
            "data_saved_success": False,
            "chat_history": [],
            "columns_to_answer": [],
            "preview_collection": None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_sidebar(self):
        """Render all sidebar components."""
        with st.sidebar:
            self._render_language_settings()
            self._render_settings()
            self._render_configuration_summary()
            
    def _render_language_settings(self):
        """Render language selection in sidebar."""
        st.header("1. Setup Language")
        language_choice = st.radio(
            "Select language:",
            [NONE, ENGLISH, VIETNAMESE],
            index=0
        )
        
        # Handle language selection
        if language_choice == ENGLISH:
            if st.session_state.get("language") != EN:
                st.session_state.language = EN
                if st.session_state.get("embedding_model_name") != 'all-MiniLM-L6-v2':
                    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    st.session_state.embedding_model_name = 'all-MiniLM-L6-v2'
                st.success("Using English embedding model: all-MiniLM-L6-v2")
        elif language_choice == VIETNAMESE:
            if st.session_state.get("language") != VI:
                st.session_state.language = VI
                if st.session_state.get("embedding_model_name") != 'keepitreal/vietnamese-sbert':
                    st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
                    st.session_state.embedding_model_name = 'keepitreal/vietnamese-sbert'
                st.success("Using Vietnamese embedding model: keepitreal/vietnamese-sbert")

    def _render_settings(self):
        """Render settings section in sidebar."""
        st.header("Settings")
        
        st.session_state.chunk_size = st.number_input(
            "Chunk Size",
            min_value=10,
            max_value=1000,
            value=st.session_state.chunk_size,
            step=10,
            help="Set the size of each chunk in terms of tokens."
        )

        st.session_state.chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=1000,
            step=10,
            value=st.session_state.chunk_size // 10,
            help="Set the overlap between chunks."
        )

        st.session_state.number_docs_retrieval = st.number_input(
            "Number of documents retrieval",
            min_value=1,
            max_value=50,
            value=st.session_state.number_docs_retrieval,
            step=1,
            help="Set the number of documents to retrieve."
        )

    def _render_configuration_summary(self):
        """Display configuration summary in sidebar."""
        st.subheader("All configurations:")
        active_collection_names = list(st.session_state.active_collections.keys())
        collection_names = ', '.join(active_collection_names) if active_collection_names else 'No collections'
        
        configs = [
            ("Active Collections", collection_names),
            ("LLM model", st.session_state.llm_name if 'llm_name' in st.session_state else 'Not selected'),
            ("Local or APIs", st.session_state.llm_type if 'llm_type' in st.session_state else 'Not specified'),
            ("Language", st.session_state.language),
            ("Embedding Model", st.session_state.embedding_model.__class__.__name__ if st.session_state.embedding_model else 'None'),
            ("Chunk Size", st.session_state.chunk_size),
            ("Number of Documents Retrieval", st.session_state.number_docs_retrieval),
            ("Data Saved", 'Yes' if st.session_state.data_saved_success else 'No'),
            ("LLM API Key Set", 'Yes' if st.session_state.get('llm_api_key') else 'No')
        ]

        for i, (key, value) in enumerate(configs, 1):
            st.markdown(f"{i}. {key}: **{value}**")

        if st.session_state.get('chunkOption'):
            st.markdown(f"10. Chunking Option: **{st.session_state.chunkOption}**")

    def render_main_content(self):
        """Render main content area of the application."""
        st.header("Peer Code Review with RAG")
        st.markdown("Design your own chatbot using the RAG system.")
        
        # Organize sections with proper numbering
        section_num = 1
        
        # Data Source Section
        self.data_processor.render_data_source_section(section_num)
        section_num += 1
        
        # LLM Setup Section
        self._render_llm_setup_section(section_num)
        section_num += 1
        
        # Search Algorithm Setup Section
        self._render_search_setup_section(section_num)
        section_num += 1
        
        # Export Section
        self._render_export_section(section_num)
        section_num += 1
        
        # Interactive Chatbot Section
        self._render_chatbot_section(section_num)
   
    def _render_llm_setup_section(self, section_num):
        """
        Render LLM setup section.
        
        Args:
            section_num (int): Section number for header
        """
        st.header(f"{section_num}. Setup LLMs")  

        st.selectbox(
            "Choose Model Source:", 
            ["Online", "Local (Ollama)"],
            index=list(self.llm_options.values()).index(self.llm_options["Local (Ollama)"]),
            key="llm_choice"
        )
    
        if st.session_state.get("llm_choice") == self.llm_options["Online"]:
            st.toast("Feature is in development", icon="✅")            
        
        elif st.session_state.get("llm_choice") == self.llm_options["Local (Ollama)"]:
            st.markdown("Please install and run Docker before running Ollama locally.")
            
        if st.button("Initialize Ollama Container"):
            with st.spinner("Setting up Ollama container..."):
                self.ollama_manager.run_ollama_container()
            
        selected_model = st.selectbox("Select a model to run", list(OLLAMA_MODEL_OPTIONS.keys()))
        real_name_model = OLLAMA_MODEL_OPTIONS[selected_model]

        if st.button("Run Selected Model"):
            local_llms = self.ollama_manager.run_ollama_model(real_name_model)
            if local_llms:
                st.session_state.llm_name = real_name_model
                st.session_state.llm_type = LOCAL_LLM
                st.session_state.local_llms = local_llms
                st.success(f"Running model: {real_name_model}")
        
    def _render_search_setup_section(self, section_num):
        """
        Render search algorithm setup section.
        
        Args:
            section_num (int): Section number for header
        """
        st.header(f"{section_num}. Set up search algorithms")
        st.radio(
            "Please select one of the options below.",
            [VECTOR_SEARCH, HYDE_SEARCH],
            captions = [
                "Search using vector similarity",
                "Search using the HYDE algorithm"
            ],
            key="search_option",
            index=0,
        )
       
    def _render_export_section(self, section_num):
        """
        Render export section.
        
        Args:
            section_num (int): Section number for header
        """
        st.header(f"{section_num}. Export Chatbot")
        if st.button("Export Chatbot"):
            self._export_chatbot()

    def _export_chatbot(self):
        """Export chatbot configuration to JSON file."""
        file_path = "pages/session_state.json"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Required fields to export
        required_fields = [
            "random_collection_name", 
            "number_docs_retrieval", 
            "embedding_model_name", 
            "llm_type", 
            "llm_name",
            "columns_to_answer",
            "search_option"
        ]

        # Check if all required fields are present in session state
        missing_fields = [field for field in required_fields if field not in st.session_state]
        if missing_fields:
            st.error(f"Missing required fields: {', '.join(missing_fields)}")
            return

        # Check if llm_type is 'local_llm'
        if st.session_state["llm_type"] != LOCAL_LLM:
            st.error("Only support exporting local LLMs.")
            return
        
        # Filter session state to only include specified fields and serializable types
        session_data = {
            key: value for key, value in st.session_state.items() 
            if key in required_fields and isinstance(value, (str, int, float, bool, list, dict))
        }

        # Save to JSON file
        with open(file_path, "w") as file:
            json.dump(session_data, file)
        
        st.success("Chatbot exported successfully!")

    def vector_search(self, model, query, active_collections, columns_to_answer, number_docs_retrieval):
        """
        Perform vector search across multiple collections.
        
        Args:
            model: The embedding model to use
            query (str): Search query
            active_collections (dict): Dictionary of active collections
            columns_to_answer (list): Columns to include in the response
            number_docs_retrieval (int): Number of results to retrieve
            
        Returns:
            tuple: (metadata list, formatted search result string)
        """
        # Validate inputs
        if not model:
            st.error("Embedding model not initialized. Please select a language first.")
            return [], ""
            
        if not active_collections:
            st.error("No collections available for search.")
            return [], ""
            
        if not columns_to_answer:
            st.error("No columns selected for answering.")
            return [], ""

        try:
            all_metadatas = []
            filtered_metadatas = []
            
            # Generate query embeddings
            try:
                query_embeddings = model.encode([query])
            except Exception as e:
                st.error(f"Error generating embeddings: {str(e)}")
                return [], ""
            
            # Search each active collection
            for collection_name, collection in active_collections.items():
                try:
                    results = collection.query(
                        query_embeddings=query_embeddings,
                        n_results=number_docs_retrieval
                    )
                    
                    if results and 'metadatas' in results and results['metadatas']:
                        # Add collection name to each metadata item
                        for meta_list in results['metadatas']:
                            for meta in meta_list:
                                meta['source_collection'] = collection_name
                        
                        # Flatten the nested metadata structure
                        for meta_list in results['metadatas']:
                            all_metadatas.extend(meta_list)
                            
                except Exception as e:
                    st.error(f"Error searching collection {collection_name}: {str(e)}")
                    continue
            
            if not all_metadatas:
                st.info("No relevant results found in any collection.")
                return [], ""
            
            # Filter metadata to only include selected columns plus source collection
            for metadata in all_metadatas:
                filtered_metadata = {
                    'source_collection': metadata.get('source_collection', 'Unknown')
                }
                for column in columns_to_answer:
                    if column in metadata:
                        filtered_metadata[column] = metadata[column]
                filtered_metadatas.append(filtered_metadata)
                
            # Format the search results
            search_result = self._format_search_results(filtered_metadatas, columns_to_answer)
            
            return [filtered_metadatas], search_result
            
        except Exception as e:
            st.error(f"Error in vector search: {str(e)}")
            return [], ""
            
    def _format_search_results(self, metadatas, columns_to_answer):
        """
        Format search results for display.
        
        Args:
            metadatas (list): List of metadata dictionaries
            columns_to_answer (list): Columns to include in the result
            
        Returns:
            str: Formatted search result string
        """
        search_result = ""
        for i, metadata in enumerate(metadatas, 1):
            search_result += f"\n{i}) Source: {metadata.get('source_collection', 'Unknown')}\n"
            for column in columns_to_answer:
                if column in metadata:
                    search_result += f"   {column.capitalize()}: {metadata.get(column)}\n"
        return search_result

    def _render_chatbot_section(self, section_num):
        """
        Render interactive chatbot section.
        
        Args:
            section_num (int): Section number for header
        """
        st.header(f"{section_num}. Interactive Chatbot")

        # Validate prerequisites 
        if not self._validate_chatbot_prerequisites():
            return

        # Allow user to select columns
        self._render_column_selector()

        # Initialize chat history if needed
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        self._display_chat_history()

        # Handle new user input
        if prompt := st.chat_input("Ask a question..."):
            self._process_user_input(prompt)
    
    def _validate_chatbot_prerequisites(self):
        """
        Validate that prerequisites for the chatbot are met.
        
        Returns:
            bool: True if prerequisites are met, False otherwise
        """
        if not st.session_state.embedding_model:
            st.error("Please select a language to initialize the embedding model.")
            return False

        if not st.session_state.active_collections:
            st.error("No collection found. Please upload data and save it first.")
            return False
            
        return True
        
    def _render_column_selector(self):
        """Render the column selector for the chatbot."""
        available_columns = list(st.session_state.chunks_df.columns)
        st.session_state.columns_to_answer = st.multiselect(
            "Select one or more columns LLMs should answer from:", 
            available_columns
        )
        
    def _display_chat_history(self):
        """Display the chat history."""
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    def _process_user_input(self, prompt):
        """
        Process a new user input in the chatbot.
        
        Args:
            prompt (str): User's prompt
        """
        # Add user message to chat history
        st.session_state.chat_history.append({"role": USER, "content": prompt})
        with st.chat_message(USER):
            st.markdown(prompt)

        with st.chat_message(ASSISTANT):
            if not st.session_state.columns_to_answer:
                st.warning("Please select columns for the chatbot to answer from.")
                return

            # Search status
            search_status = st.empty()
            search_status.info("Searching for relevant information...")
            
            try:
                # Perform vector search
                metadatas, retrieved_data = self.vector_search(
                    st.session_state.embedding_model,
                    prompt,
                    st.session_state.active_collections,
                    st.session_state.columns_to_answer,
                    st.session_state.number_docs_retrieval
                )
                
                if not metadatas or not retrieved_data:
                    search_status.warning("No relevant information found.")
                    return
                    
                enhanced_prompt = f"""The prompt of the user is: "{prompt}". Answer it based on the following retrieved data: \n{retrieved_data}"""
                
                # Display search results
                search_status.success("Found relevant information!")
                    
                # Show retrieved data in sidebar
                st.sidebar.subheader("Retrieved Data")
                if metadatas and metadatas[0]:
                    st.sidebar.dataframe(pd.DataFrame(metadatas[0]))
                
                # Call LLM for response
                try:
                    response = st.session_state.local_llms.generate_content(enhanced_prompt)
                    search_status.empty()
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": ASSISTANT, "content": response})
                except Exception as e:
                    search_status.error(f"Error from LLM: {str(e)}")
               
            except Exception as e:
                search_status.error(f"An error occurred: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    def get_llm_options(self):
        """
        Return LLM options dictionary.
        
        Returns:
            dict: Dictionary of LLM options
        """
        return self.llm_options