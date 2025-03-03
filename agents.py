#!/usr/bin/env python
# coding: utf-8
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

class LLMPrompt:
    def __init__(self, role, prompt_type, prompt_message):
        self.message = {
            "role": role,
            "content": [{
                "type": prompt_type,
                prompt_type: prompt_message
            }]
        }
            
class LLMAgentSystemPrompt(LLMPrompt):
    def __init__(self, system_prompt):
        super().__init__(role="developer", prompt_type="text", prompt_message=system_prompt)


class LLMClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def call(self, model, messages, max_tokens=1000):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

class LLMAgent:
    def __init__(self, system_prompt, model, api_key):
        self.client = LLMClient(api_key=api_key)
        self.model = model
        self.system_prompt = LLMAgentSystemPrompt(system_prompt).message



class Chainer(LLMAgent):
    def __init__(self, system_prompt, steps, inp, model, api_key):
        super().__init__(system_prompt, model, api_key)        
        self.steps = steps
        self.inp = inp
        self.model = model

    def chain(self, verbose=False):
        chain = [self.system_prompt]
        current_input = self.inp
        for ix, step in enumerate(self.steps):            
            new_step = LLMPrompt("user", "text", f"{step}\nInput: {current_input}").message
            chain.append(new_step)
            response_content = self.client.call(
                model=self.model,
                messages=chain,
                max_tokens=1000)

            # Add the assistant's response to the chain
            assistant_response = LLMPrompt("assistant", "text", response_content).message
            chain.append(assistant_response)
            if verbose:
                print(f"STEP: {ix}")
                print(response_content)
                print("="*36)
            current_input = response_content
        return response_content


class Blogger(Chainer):
    def __init__(self, topic, target_audience, word_count, model="gpt-4o-mini", api_key=None):
        # Define the system prompt for blog post generation
        system_prompt = """You are a professional blog content creator who specializes in creating
        high-quality, engaging blog posts on various topics. You follow SEO best practices
        and create content that is both informative and engaging."""
        
        # Define the steps for the blog post creation pipeline
        steps = [
            # Step 1: Generate an outline
            """Create a detailed outline for a blog post on the given topic.
            Include a title, introduction, 4-6 main sections with subpoints, and conclusion.
            Format the outline with clear headings and bullet points.""",
            
            # Step 2: Validate and refine the outline
            """Review the outline and ensure it meets these criteria:
            - Covers the topic comprehensively
            - Flows logically from point to point
            - Addresses the needs of the target audience
            - Includes specific, actionable information
            If any criteria are not met, refine the outline accordingly.""",
            
            # Step 3: Expand the outline into a complete post
            """Based on the validated outline, write the complete blog post.
            Include all of the following:
            1. An engaging title
            2. A hook-filled introduction
            3. Well-developed main sections with subheadings
            4. A strong conclusion with a call-to-action
            5. Write in a conversational but professional tone    
            The complete post should be approximately {word_count} words."""
        ]
        
        
        # Prepare input for the chaining process
        input_data = f"""
        Topic: {topic}
        Target Audience: {target_audience}
        Approximate Word Count: {word_count}
        """
        # Initialize the parent class
        super().__init__(system_prompt, steps, input_data, model, api_key)
    
    def generate_blog_post(self, verbose=True):
        """Generate a complete blog post using the chaining process"""
        print(f"🚀 Starting blog post generation process...")
        result = self.chain(verbose=verbose)
        print(f"✅ Blog post generation complete!")
        return result
    
    def save_to_file(self, filename):
        """Save the generated blog post to a markdown file"""
        blog_post = self.generate_blog_post(verbose=False)
        with open(filename, 'w') as f:
            f.write(blog_post)
        print(f"📝 Blog post saved to {filename}")


class Router(LLMAgent):
    def __init__(self, system_prompt, choices, model, api_key, input_data):
        super().__init__(system_prompt, model, api_key)
        self.input_data = input_data
        self.choices = choices
    
    def route(self):
        routing_prompt_text=f"Analyze the following data {self.input_data.strip()} and based on the data select the most appropriate of the following options: {self.choices}"
        self.routing_prompt = LLMPrompt("user", "text", routing_prompt_text)
        chain = [self.system_prompt, self.routing_prompt.message]
        response = self.client.call(
            model=self.model,
            messages=chain,)
        return response

### Parallelization: Distributes independent subtasks across multiple LLMs for concurrent processing


class ParallelAgent(LLMAgent):
    def __init__(self, system_prompt, tasks, instruction, model, api_key):
        self.tasks = tasks
        self.instruction = instruction
        super().__init__(system_prompt, model, api_key)

    def run_parallel(self, n_workers=3):    
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            f = lambda task : self.client.call(
                model=self.model,
                max_completion_tokens=1000,
                messages = [self.system_prompt, LLMPrompt("user", "text", f"{self.instruction}\nInput: {task}").message])
            futures = [executor.submit(f, f"{self.instruction}\nInput: {x}") for x in self.tasks]
        return [f.result() for f in futures]
