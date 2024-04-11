from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
import sys


ollama_gemma = Ollama(model='gemma:7b')
ollama_mistral = Ollama(model='mistral:7b')

def write_blog(topic):
  # Create Researcher agent
  researcher = Agent(
      role='Senior Research Analyst',
      goal=f"Uncover research in {topic}",
      backstory="""You work at a leading research company.
      Your expertise lies in identifying emerging trends, research and discover.
      You have a knack for dissecting complex data and presenting actionable insights.""",
      verbose=True,
      allow_delegation=False,
      llm=ollama_gemma
  )

  # Create Writer agent
  writer = Agent(
      role='Content Strategist',
      goal=f"Craft compelling content on {topic}",
      backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
      You transform complex concepts into compelling narratives.""",
      verbose=True,
      allow_delegation=False,
      llm=ollama_mistral
  )

  # Create tasks for your agents
  task1 = Task(
    description=f"Conduct a comprehensive analysis of {topic}.",
    expected_output="Full analysis report in bullet points",
    agent=researcher
  )

  task2 = Task(
    description=f"""Using the insights provided, develop an engaging blog
    post that highlights the topic of {topic}.
    Your post should be informative yet accessible, catering to any audience, easy to understand.
    Make it sound cool, avoid complex words so it doesn't sound like AI.""",
    expected_output="Full blog post of at least 4 paragraphs",
    agent=writer
  )

  # Instantiate your crew with a sequential process
  crew = Crew(
      agents=[researcher, writer],
      tasks=[task1, task2],
      verbose=2, # You can set it to 1 or 2 to different logging levels
  )

  # Get your crew to work!
  result = crew.kickoff()
  return result

def main():
  if len(sys.argv) < 2:
    print("Please provide a topic as an argument.")
    print('Usage: python blog.py "AI and data science trends in 2024"')
    sys.exit(1)

  # Get the first command-line argument after the script name
  topic = sys.argv[1]

  # Now you can use 'argument' in your script
  print("\n\n#### Topic ####")
  print(topic)

  if topic == '':
    print("Topic is empty.")
    sys.exit(1)

  result = write_blog(topic)

  print("\n\n#### Result ####")
  print(result)

if __name__ == '__main__':
  main()
