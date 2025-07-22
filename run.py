import os
import json
import openai
from agent import AnalysisAgent
# from notebook_generator import generate_notebook
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Agent for generate new hypothesis")
    parser.add_argument(
        "--skip_novelty_check",
        action="store_true",
        default=False,
        help="Skip novelty check and use existing ideas",
    )
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--analysis_name",
        type=str,
        default="test",
        help="input to run AI Scientist on.",
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="data path to use for the agent.",
    )
    
    parser.add_argument(
        "--paper_path",
        type=str,
        default=None,
        help="paper path to use for the agent.",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="o3-mini",
        # choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    # parser.add_argument(
    #     "--correlative_papers",
    #     type=str,
    #     default=None,
    #     help="correlative recent papers related to the input to use for the agent.",
    # )
    # parser.add_argument(
    #     "--field_of_study",
    #     type=str,
    #     default=None,
    #     help="Field of study to use for the agent.",
    # )
    parser.add_argument(
        "--skip_human_response",
        action="store_true",
        default=False,
        help="skip human response.",
    )

    parser.add_argument(
        "--num_ideas",
        type=int,
        default=3,
        help="Number of ideas to generate",
    )
    parser.add_argument(
        "--num_critique_steps",
        type=int,
        default=3,
        help="Number of analyses to run",
    )
    parser.add_argument(
        "--num_reflections",
        type=int,
        default=3,
        help="Number of reflections to use",
    )
    parser.add_argument(
        "--run_comparison",
        action="store_true",
        default=False,
        help="Run both agent analysis and normal LLM analysis for comparison",
    )
    return parser.parse_args()

# Initialize the agent
if __name__ == "__main__":
    args = parse_arguments()
    
    agent = AnalysisAgent(
        skip_novelty_check=args.skip_novelty_check,
        analysis_name=args.analysis_name,
        data_path=args.data_path,
        paper_path=args.paper_path,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model_name=args.model_name,
        # correlative_papers=args.correlative_papers,
        # knowledge_path=os.path.join(os.getcwd(), "knowledge_background/molecular_gen_data_0_to_15.csv"),
        # field_of_study=args.field_of_study,
        skip_human_response=args.skip_human_response,
        num_ideas=args.num_ideas,
        num_critique_steps=args.num_critique_steps,
        num_reflections=args.num_reflections
    )

# Run the analysis
if args.run_comparison:
    print("Running comparison analysis...")
    agent.run_comparison_analysis()  # This will run both agent and normal LLM analysis
else:
    agent.run()  # This will run all the analyses the agent decides to attempt.