import openai
import os
import json
import scanpy as sc
import nbformat as nbf
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor
import tempfile
import numpy as np
import gc
import datetime
from logger import Logger
import base64
import h5py
from h5py import Dataset, Group
import re
import shutil
from PIL import Image
import io
import PyPDF2
import requests
import csv
from llm import get_response_from_llm, extract_json_between_markers, AVAILABLE_LLMS
import os.path as osp

# AVAILABLE_PACKAGES = "scanpy, scvi, CellTypist, anndata, matplotlib, numpy, seaborn, pandas, scipy"
class AnalysisAgent:
    def __init__(self, skip_novelty_check, analysis_name, data_path, paper_path, openai_api_key, model_name, skip_human_response = False, num_ideas = 3, num_critique_steps=3, num_reflections=3, prompt_dir="prompts"):
        # self.skip_novelty_check = skip_novelty_check
        self.analysis_name = analysis_name
        self.data_path = data_path
        self.paper_path = paper_path
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        # self.correlative_papers = correlative_papers
        # self.field_of_study = field_of_study
        self.skip_human_response = skip_human_response
        self.num_ideas = num_ideas
        self.num_critique_steps = num_critique_steps
        self.num_reflections = num_reflections
        self.prompt_dir = prompt_dir
        # self.knowledge_path = knowledge_path
        self.output_dir = os.path.join("outputs", f"{analysis_name}")
        self.client = openai.OpenAI(api_key=openai_api_key)
        os.makedirs(self.output_dir, exist_ok=True)
        # analyses_overview = open(os.path.join(self.prompt_dir, "DeepResearch_Analyses.txt")).read()
        # Load coding system prompt from JSON file
        with open(os.path.join(self.prompt_dir, "coding_system_prompt.json"), "r") as f:
            coding_prompt_data = json.load(f)
        self.coding_system_prompt = coding_prompt_data
        self.logger = Logger(self.analysis_name)
        self.logger.log_action(
            "Agent initialized", 
            f"data_path: {data_path}\n" +
            f"model: {model_name}\n" +
            f"max_iterations: {num_critique_steps}"
        )
        
        
        if self.data_path == "":
            self.data_summary = ""
        else:
            print("Loading data for summarization...")
            self.data_frame = self.load_data(self.data_path)
            self.data_summary = self.summarize_data()
            self.logger.log_action("Data loaded and summarized", self.data_summary)
            print(f"Loaded {self.data_path} and summarized")
        # end
        # if self.knowledge_path is None:
        #     self.knowledge = self.generate_background_knowledge(field_of_study)
        # else:
        #     self.knowledge = self.load_knowledge(self.knowledge_path)
        self.paper = self.load_paper(self.paper_path)
        self.paper_summary = self.summarize_paper_with_llm(self.paper)
        
    def load_data(self, data_path):
        """Load data from a CSV file into a pandas DataFrame"""
        try:
            df = pd.read_csv(data_path)
            print(f"Loaded CSV data: {len(df)} rows × {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return None

    def summarize_data(self, length_cutoff=10):
        """
        Summarize the data in the DataFrame

        Args:
            length_cutoff (int): How many max unique values to include for each column
        """
        if self.data_frame is None:
            return "No data loaded"
            
        summarization_str = f"Below is a description of the columns in the dataset: \n"
        columns = self.data_frame.columns
        
        for col in columns:
            # Get unique values, handling different data types
            if pd.api.types.is_numeric_dtype(self.data_frame[col]):
                # For numeric columns, show min, max, and mean
                min_val = self.data_frame[col].min()
                max_val = self.data_frame[col].max()
                mean_val = self.data_frame[col].mean()
                summarization_str += f"Column {col} (numeric): min={min_val}, max={max_val}, mean={mean_val}\n"
            else:
                # For categorical/string columns, show unique values
                unique_vals = self.data_frame[col].unique()
                summarization_str += f"Column {col} contains the values {unique_vals[:length_cutoff]}\n"
                
        return summarization_str
    
    def load_knowledge(self, knowledge_path):
        """
        Load knowledge about molecular generation from CSV file
        
        Args:
            knowledge_path (str): Path to the CSV file containing molecular generation knowledge
        """
        if knowledge_path is None:
            return "No knowledge file provided"
            
        try:
            # Read the CSV file
            df = pd.read_csv(knowledge_path)
            
            # Create a structured knowledge string
            knowledge_str = "Molecular Generation Backgrounds, Methods and Approaches:\n\n"
            
            # Process each row to create a comprehensive knowledge base
            for _, row in df.iterrows():
                # Get the context (paper title/description)
                context = row['context']
                
                # Get the analysis titles and full descriptions
                analyses_titles = eval(row['analyses_titles'])
                analyses_full = eval(row['analyses_full'])
                
                # Add to knowledge string
                knowledge_str += f"Background: {context}\n"
                knowledge_str += "Key Components:\n"
                
                # Add each analysis component with its description
                for method in analyses_titles:
                    knowledge_str += f"- {method}\n"
                knowledge_str += "\n"
                for analysis in analyses_full:
                    knowledge_str += f"- {analysis['title']}: {analysis['description']}\n"
                knowledge_str += "\n"
            
            self.logger.log_action("Knowledge loaded", f"Loaded knowledge from {knowledge_path}")
            return knowledge_str
            
        except Exception as e:
            print(f"Error loading knowledge file: {str(e)}")
            return "No knowledge loaded"
        
    # def generate_background_knowledge(self, field_of_study):
    #     # TODO: generate background knowledge based on field of study, the study famous papers or popular papers
    #     query = f"{field_of_study} famous papers or famous papers or top cited papers"
    #     search = GoogleSerperAPIWrapper(api_key=os.getenv("SERPER_API_KEY"))
    #     results = search.run(query)
    #     # print(f"Results: {results}")
        
    #     data = []
    #     if 'scholar' in results:
    #         papers = results['scholar']
    #     elif 'organic' in results:
    #         papers = results['organic']
    #     else:
    #         papers = results.get('results')
            
    #     for idx, paper in enumerate(papers):
    #         title = paper.get('title')
    #         snippet = paper.get('snippet')
    #         link = paper.get('link')
    #         analyses_titles, analyses_full = self.extract_paper_analysis(title, snippet, link)
    #         data.append({
    #             'id': idx,
    #             'context': title,
    #             'analyses_titles': analyses_titles,
    #             'analyses_full': analyses_full
    #         })
        
    #     # Print the data as JSON for debugging
    #     print("\n" + "="*50)
    #     print("DEBUG: Generated Background Knowledge Data (JSON format):")
    #     print("="*50)
    #     print(json.dumps(data, indent=2, ensure_ascii=False))
    #     print("="*50)
        
    #     output_dir = "knowledge_background"
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     filename = os.path.join(output_dir, f"{field_of_study.replace(' ', '_')}_background_knowledge.csv")
    #     with open(filename, "w", newline='', encoding='utf-8') as csvfile:
    #         fieldnames = ["id", "context", "analyses_titles", "analyses_full"]
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for row in data:
    #             writer.writerow(row)
    #     return filename
    # def generate_background_knowledge(self, field_of_study, api_key):
    #     api = GoogleSerperAPIWrapper(api_key)
    #     query = f"site:scholar.google.com {field_of_study} famous papers OR recent papers"
    #     results = api.search(query)
    #     papers = []
    #     idx = 0
    #     for item in results.get("organic", []):
    #         title = item.get("title", "")
    #         snippet = item.get("snippet", "")
    #         link = item.get("link", "")
    #         # Filter for likely papers
    #         if any(domain in link for domain in ["scholar.google.com", "arxiv.org", "pubmed", "nature.com", "sciencedirect.com"]):
    #             analyses_titles = extract_key_topics(snippet or title)
    #             analyses_full = [{"title": t, "description": f"Contribution related to {t}."} for t in analyses_titles]
    #             papers.append({
    #                 "id": idx,
    #                 "context": title,
    #                 "analyses_titles": str(analyses_titles),
    #                 "analyses_full": str(analyses_full)
    #             })
    #             idx += 1
    #     # Save to CSV
    #     out_dir = "yuesu/knowledge_background"
    #     os.makedirs(out_dir, exist_ok=True)
    #     filename = os.path.join(out_dir, f"{field_of_study.replace(' ', '_')}_background_knowledge.csv")
    #     with open(filename, "w", newline='', encoding='utf-8') as csvfile:
    #         fieldnames = ["id", "context", "analyses_titles", "analyses_full"]
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for row in papers:
    #             writer.writerow(row)
    #     return filename
    def extract_paper_analysis(self, paper_title, paper_snippet, paper_link=None):
        """
        Extract key methods and contributions from a paper using LLM
        
        Args:
            paper_title (str): Title of the paper
            paper_snippet (str): Abstract or snippet of the paper
            paper_link (str): Link to the paper (optional)
        
        Returns:
            tuple: (analyses_titles, analyses_full)
        """
        try:
            # Create a comprehensive prompt for analysis
            prompt = f"""Analyze the following scientific paper and extract the key methods and contributions:

Title: {paper_title}
Abstract/Snippet: {paper_snippet}
Link: {paper_link if paper_link else print("invalid")}

Please identify 3-5 key methods, techniques, or contributions from this paper. For each one, provide:
1. A short title (2-4 words)
2. A brief description of what the method/contribution does

Format your response as a JSON object with this structure:
{{
    "analyses_titles": ["Method 1", "Method 2", "Method 3"],
    "analyses_full": [
        {{"title": "Method 1", "description": "Description of method 1"}},
        {{"title": "Method 2", "description": "Description of method 2"}},
        {{"title": "Method 3", "description": "Description of method 3"}}
    ]
}}

Focus on:
- Novel algorithms or techniques
- Key methodological innovations
- Important contributions to the field
- Unique approaches or frameworks
"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing scientific papers and extracting key methods and contributions. Always respond with valid JSON format."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            # Clean up the response (remove markdown formatting if present)
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(content)
                analyses_titles = result.get('analyses_titles', [])
                analyses_full = result.get('analyses_full', [])
                
                # Ensure we have valid data
                if not analyses_titles or not analyses_full:
                    # Fallback to simple extraction
                    print(f"Failed to extract paper analysis: {content}")
                    # analyses_titles = [f"Key contribution from {paper_title[:20]}..."]
                    # analyses_full = [{"title": analyses_titles[0], "description": paper_snippet[:200] + "..."}]
                
                return analyses_titles, analyses_full
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                print(f"Failed to parse JSON response: {content}")
            
        except Exception as e:
            print(f"Error extracting paper analysis: {str(e)}")
            # Fallback to simple extraction
    
    def load_paper(self, paper_path):
        """
        Load and process paper summary from a PDF file
        
        Args:
            paper_path (str): Path to the paper PDF file
        """
        try:
            # Check if file is PDF
            if not paper_path.lower().endswith('.pdf'):
                raise ValueError("File must be in PDF format")
                
            # Extract text from PDF
            paper_text = ""
            with open(paper_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                for page in pdf_reader.pages:
                    paper_text += page.extract_text() + "\n"
            
            # Basic processing of the paper text
            # Remove extra whitespace and normalize line endings
            paper_text = ' '.join(paper_text.split())
            
            # Store the processed paper text
            # self.paper_summary = paper_text
            
            # Log the action
            self.logger.log_action("Paper summary loaded", f"Loaded PDF from {paper_path}")
            
            return paper_text
            
        except Exception as e:
            print(f"Error loading PDF paper: {str(e)}")
            return "No paper summary loaded"
            
    def summarize_paper_with_llm(self, paper_text):
        """
        Use LLM to generate a structured summary of the paper
        
        Args:
            paper_text (str): The full text of the paper
        """
        try:
            prompt = f"""Please provide a structured summary of the following paper:

{paper_text}

Please structure the summary as follows:
1. Main Objective
2. Main ideas and hypotheses
3. Key Methods
4. Important Findings
5. Significance results
5. Limitations
6. Future Work
"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing scientific papers, and you are given a paper. Please summarize the paper based on the its idea, title, methods, and results."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating LLM summary: {str(e)}")
            return paper_text
    
    def generate_analysis(self, max_num_generations=5, num_reflections=3, skip_human_response=False):
        previous_ideas = []
        prompt_first_draft = open(os.path.join(self.prompt_dir, "first_draft.txt")).read()
        prompt_first_draft = prompt_first_draft.format(
            # scientific_background=self.knowledge, 
            data_summary=self.data_summary, 
            previous_ideas=[], 
            paper_txt=self.paper_summary,
            num_reflections=num_reflections,
        )
        prompt_reflection = open(os.path.join(self.prompt_dir, "reflection.txt")).read()
        
        self.logger.log_prompt("user", prompt_first_draft, "Initial Analysis")
        for i in range(max_num_generations):
            print(f"Generating idea {i+1}/{max_num_generations}")
            msg_history = []
            response, msg_history = get_response_from_llm(
                prompt_first_draft,
                client=self.client,
                model=self.model_name,
                system_message=self.coding_system_prompt["system_prompt"],
                msg_history=msg_history
            )
            json_output = extract_json_between_markers(response)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    response, msg_history = get_response_from_llm(
                        prompt_reflection.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=self.client,
                        model=self.model_name,
                        system_message=self.coding_system_prompt["self_reflection_system_prompt"],
                        msg_history=msg_history
                    )
                    json_output = extract_json_between_markers(response)
                    assert json_output is not None, "Failed to extract JSON from LLM output"
                    print(json_output)
            if not skip_human_response:
                final_analysis, msg_history = self.get_feedback(self.num_critique_steps, json_output, msg_history)
                if final_analysis == "bad":
                    continue
            previous_ideas.append(json.dumps(json_output))
        ideas = []
        for idea_str in previous_ideas:
            ideas.append(json.loads(idea_str))
        with open(osp.join(self.output_dir, "ideas.json"), "w") as f:
            json.dump(ideas, f, indent=4)
        return ideas
    
    # def search_for_papers(query, result_limit=10, engine="semanticscholar") -> Union[None, List[Dict]]:
    #     if not query:
    #         return None
    #     if engine == "semanticscholar":
    #         rsp = requests.get(
    #             "https://api.semanticscholar.org/graph/v1/paper/search",
    #             headers={"X-API-KEY": S2_API_KEY} if S2_API_KEY else {},
    #             params={
    #                 "query": query,
    #                 "limit": result_limit,
    #                 "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
    #             },
    #         )
    #         print(f"Response Status Code: {rsp.status_code}")
    #         print(
    #             f"Response Content: {rsp.text[:500]}"
    #         )  # Print the first 500 characters of the response content
    #         rsp.raise_for_status()
    #         results = rsp.json()
    #         total = results["total"]
    #         time.sleep(1.0)
    #         if not total:
    #             return None

    #         papers = results["data"]
    #         return papers
    # def go_deep_research(self, analysises):
    #     for analysis in analysises:
            
    #     pass
                
                    
                    
                    
                    

    # def create_ideas(self):
    #     past_analyses = ""
    #     analyses = []
    #     for analysis_idx in range(self.num_analyses):
    #         print(f"\n🚀 Starting Analysis {analysis_idx+1}")

    #         analysis = self.generate_initial_analysis(past_analyses)

    #         # modified_analysis = self.get_feedback(analysis, past_analyses, None)
    #         # summary = modified_analysis["summary"]

    #         past_analyses += f"{analysis['hypothesis']}\n"
    #         analyses.append()

    #     return analyses
    def critique_step(self, analysis, msg_history):
        # Get human feedback
        human_response = self.get_human_response(analysis)
        if human_response == "bad":
            return "bad", msg_history
        elif human_response == "satisfied":
            return "satisfied", msg_history
        
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        
        prompt = open(os.path.join(self.prompt_dir, "critic.txt")).read()
        prompt = prompt.format(
            hypothesis=hypothesis,
            analysis_plan=analysis_plan,
            human_response=human_response,
            paper_summary=self.paper_summary,
        )
        
        # Add instruction for JSON response
        
        
        response, msg_history = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model_name,
            system_message=self.coding_system_prompt["critic_system_prompt"],
            msg_history=msg_history
        )
        json_output = extract_json_between_markers(response)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        print(json_output)
        return json_output, msg_history
    
    def get_feedback(self, num_critique_steps, analysis, msg_history):
        for i in range(num_critique_steps):
            modified_analysis, msg_history = self.critique_step(analysis, msg_history)
            if modified_analysis == "bad":
                return modified_analysis, msg_history
            elif modified_analysis == "satisfied":
                return analysis, msg_history
            else:
                analysis = modified_analysis
        return modified_analysis, msg_history
    
    def get_human_response(self, analysis):
        """Get direct feedback from human user in terminal"""
        print("\n" + "="*50)
        print("Please provide your feedback on the current analysis")
        print("="*50)
        print("\nCurrent Analysis:")
        print(f"\nHypothesis: {analysis['hypothesis']}")
        print("\nAnalysis Plan:")
        for step in analysis['analysis_plan']:
            print(f"- {step}")
        print(f"\nInterestingness: {analysis['interestingness']}")
        print(f"\nFeasibility: {analysis['feasibility']}")
        print(f"\nNovelty: {analysis['novelty']}")
        print("\n" + "-"*50)
        print("Options:")
        print("1. Press Enter when done with feedback")
        print("2. Type 'bad' to indicate that you think the topic is not interesting and you want to skip and get the next one")
        print("3. Type 'satisfied' if you're happy with the current plan")
        print("-"*50)
        print("\nEnter your feedback or option:")
        
        try:
            line = input()
            if not line:  # Single Enter press
                human_response = ""
            elif line.lower() == 'bad':
                human_response = "bad"
            elif line.lower() == 'satisfied':
                human_response = "satisfied"
            else:
                human_response = line
        except EOFError:
            human_response = ""
        
        self.logger.log_action("Human feedback received", human_response)
        # def get_input():
        #     print("Please enter your feedback on the hypothesis and analysis plan. Enter 'q' to quit.")
        #     line = input()
        #     return line
        # from langchain_community.tools import HumanInputRun
        # human_feedback = HumanInputRun(input_func=get_input, description="Use this tool to obtain feedback on the hypothesis and analysis plan.", name="Human Feedback")
        return human_response
    # def search_for_papers(self, query):
    #     search = GoogleSerperAPIWrapper(api_key=os.getenv("SERPER_API_KEY"))
    #     results = search.run(query)
    #     return results
    
    def check_novelty(self, analysis):
        novelty_prompt = open(os.path.join(self.prompt_dir, "novelty.txt")).read()
        novelty_prompt = novelty_prompt.format(
            hypothesis=analysis["hypothesis"],
            analysis_plan=analysis["analysis_plan"],
            paper_summary=self.paper_summary,
        )
        response, msg_history = get_response_from_llm(
            msg=novelty_prompt,
            client=self.client,
            model=self.model_name,
            system_message=self.coding_system_prompt["novelty_system_prompt"],
            msg_history=msg_history
        )
        return response
    
    def run(self):
            # Initial analysis from LLM
        analysis = self.generate_analysis(self.num_ideas, self.num_reflections, self.skip_human_response)
        print(analysis)
            # hypothesis = analysis["hypothesis"]
            # analysis_plan = analysis["analysis_plan"]
            # rationale = analysis["rationale"]
            
            # # Display current analysis for human review
            # print("\nCurrent Analysis:")
            # print(f"\nHypothesis: {hypothesis}")
            # print("\nAnalysis Plan:")
            # for step in analysis_plan:
            #     print(f"- {step}")
            # print(f"\nRationale: {rationale}")
            # Get feedback and incorporate it (this will handle the iterations internally)
            
            # Log final analysis

    def normal_analysis_with_llm(self):
        """Perform a simple analysis using ChatGPT without the agent's structured approach"""
        print("="*50)
        print("RUNNING NORMAL LLM ANALYSIS (BASELINE)")
        print("="*50)
        
        # Load and summarize data
        if self.data_path == "":
            data_summary = "No data provided"
        else:
            data_summary = self.summarize_data()
        
        # Load and summarize paper
        paper_text = self.load_paper(self.paper_path)
        paper_summary = self.summarize_paper_with_llm(paper_text)
        
        # Create a simple prompt for direct LLM analysis
        simple_prompt = f"""
You are a research analyst. Given the following information, provide a research hypothesis and analysis plan.

DATA SUMMARY:
{data_summary}

PAPER SUMMARY:
{paper_summary}

Please provide:
1. A specific hypothesis about what new insights could be discovered based on the data and paper
2. A step-by-step analysis plan to test this hypothesis
3. Rate the interestingness (0-10), feasibility (0-10), and novelty (0-10) of your proposed analysis

Respond in the following JSON format:
{{
    "hypothesis": "Your specific hypothesis here",
    "analysis_plan": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
    "interestingness": "0-10",
    "feasibility": "0-10",
    "novelty": "0-10",
}}
"""
        
        # Get response from LLM
        response, _ = get_response_from_llm(
            msg=simple_prompt,
            client=self.client,
            model=self.model_name,
            system_message="You are a skilled research analyst. Always respond with valid JSON format.",
            msg_history=[]
        )
        
        # Extract JSON response
        json_output = extract_json_between_markers(response)
        if json_output is None:
            print("Failed to extract JSON from LLM response")
            return None
        
        print("\nNORMAL LLM ANALYSIS RESULT:")
        print(json.dumps(json_output, indent=2))
        
        # Save the result
        with open(osp.join(self.output_dir, "normal_llm_analysis.json"), "w") as f:
            json.dump(json_output, f, indent=4)
        
        self.logger.log_action("Normal LLM analysis completed", json.dumps(json_output))
        
        return json_output
    
    def run_comparison_analysis(self):
        """Run both the agent analysis and normal LLM analysis for comparison"""
        print("="*60)
        print("RUNNING COMPARISON ANALYSIS")
        print("="*60)
        
        # Run normal LLM analysis first
        normal_result = self.normal_analysis_with_llm()
        
        print("\n" + "="*60)
        print("RUNNING AGENT ANALYSIS")
        print("="*60)
        
        # Run agent analysis
        agent_result = self.generate_analysis(self.num_ideas, self.num_reflections, self.skip_human_response)
        
        # Compare results
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        if normal_result and agent_result:
            print("\nNORMAL LLM ANALYSIS:")
            print(f"Hypothesis: {normal_result.get('hypothesis', 'N/A')[:100]}...")
            print(f"Interestingness: {normal_result.get('interestingness', 'N/A')}")
            print(f"Feasibility: {normal_result.get('feasibility', 'N/A')}")
            print(f"Novelty: {normal_result.get('novelty', 'N/A')}")
            
            print(f"\nAGENT ANALYSIS (Generated {len(agent_result)} ideas):")
            for i, idea in enumerate(agent_result):
                print(f"\nIdea {i+1}:")
                print(f"Hypothesis: {idea.get('hypothesis', 'N/A')[:100]}...")
                print(f"Interestingness: {idea.get('interestingness', 'N/A')}")
                print(f"Feasibility: {idea.get('feasibility', 'N/A')}")
                print(f"Novelty: {idea.get('novelty', 'N/A')}")
        
        # Save comparison results
        comparison_data = {
            "normal_llm_analysis": normal_result,
            "agent_analysis": agent_result,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(osp.join(self.output_dir, "comparison_results.json"), "w") as f:
            json.dump(comparison_data, f, indent=4)
        
        self.logger.log_action("Comparison analysis completed", "Both analyses saved to output directory")
        
        return comparison_data
    
    
# class GoogleSerperAPIWrapper:
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.base_url = "https://google.serper.dev/scholar"
    
#     def run(self, query, num_results=10):
#         headers = {
#             "X-API-KEY": self.api_key,
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "q": query,
#             "num": num_results
#         }
#         response = requests.post(self.base_url, headers=headers, json=payload)
#         response.raise_for_status()
#         return response.json()