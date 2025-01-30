from crewai import Agent, Task, Crew, Process,LLM
from dotenv import load_dotenv
from crewai_tools import SerperDevTool
import os

# Load environment variables
load_dotenv()

def compare_agent(product_name):
    # Initialize the SerperDev search tool
    web_search_tool = SerperDevTool(n_results=5)

    # Initialize custom LLM configuration
    llm = LLM(
        model="ollama/llama3.2",
        base_url="http://localhost:11434"
    )

    research_agent = Agent(
        role='Research Agent',
        goal='Find accurate price information for healthcare products',
        backstory="""You are an expert at finding and comparing healthcare product prices
        across different online stores. You're thorough and always verify information.""",
        tools=[web_search_tool],
        llm=llm,
        verbose=True
    )
    
    analysis_agent = Agent(
        role='Analysis Agent',
        goal='Analyze and compare prices from different sources',
        backstory="""You are an expert at analyzing price data and presenting it in a clear,
        actionable format. You understand healthcare product pricing patterns.""",
        llm=llm,
        tools=[web_search_tool],
        verbose=True
    )

    # research_agent = Agent(
    #         role='Research Agent',
    #         goal='Find accurate price information quickly',
    #         backstory="""Expert at rapid price comparison across healthcare e-commerce sites.
    #         Focus on essential information only.""",
    #         tools=[web_search_tool],
    #         llm=llm,
    #         verbose=True  # False if u wanna Reduce logging for speed
    #     )
        
    # analysis_agent = Agent(
    #         role='Analysis Agent',
    #         goal='Quick price analysis and comparison',
    #         backstory="""Efficient at analyzing price data and providing concise summaries.
    #         Focus on key metrics only.""",
    #         llm=llm,
    #         tools=[web_search_tool],
    #         verbose=True
    # )

    research_task = Task(
        description=f"""Quickly find prices for {product_name} from major healthcare 
        e-commerce sites. Focus on top 5 most relevant results.""",
        expected_output="""Concise list of prices:
        - Store names
        - Prices in INR
        - Direct product URLs""",
        agent=research_agent
    )

    analysis_task = Task(
            description="""Quick analysis of price data.Verify accuracy and remove any outliers or suspicious entries. Format the data consistently.Focus on essential metrics only.""",
            expected_output="""Brief price analysis:
            - Price range (lowest to highest)
            - Best value option
            - Top recommendation""",
            agent=analysis_agent
    )

    # Create crew and run tasks
    crew = Crew(
        agents=[research_agent, analysis_agent],
        tasks=[research_task, analysis_task],
        process=Process.sequential
    )

    result = crew.kickoff()
    return result

# Example usage
if __name__ == "__main__":
    try:
        result = compare_agent("azoran")
        print("\nPrice Comparison Results:")
        print(result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")