# Databricks notebook source
# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC # <font color="#76b900">Running State Chains</font>
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC In the previous notebook, we introduced some key LangChain Expression Language (LCEL) material regarding runnables. By now, you should be comfortable with both internal and external reasoning, as well as how to develop pipelines that facilitate it! In this notebook, we will make our way towards more advanced paradigms that will allow us to orchestrate more complex dialog management strategies and begin to execute on longer-form document reasoning.
# MAGIC <br>
# MAGIC
# MAGIC ### **Learning Objectives:**
# MAGIC
# MAGIC - Learning how to leverage runnables to orchestrate interesting LLM systems.  
# MAGIC - Understanding how running state chains can be used for dialog management and iterative decision-making.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ### **Questions To Think About:**
# MAGIC
# MAGIC - Would there ever be a use for a single-module variant of the running state chain that is not constantly querying the environment for input?
# MAGIC - You may notice that the JSON prediction is actually working pretty well. It might not always work so well depending on the questions and the JSON format complexity. What kinds of issues do you expect to encounter in this regard?
# MAGIC - What kinds of approaches can you think of completely swapping prompts as part of the running state chain?
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ### **Environment Setup:**

# COMMAND ----------

# MAGIC %pip install -q langchain langchain-nvidia-ai-endpoints rich
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

import os
os.environ["NVIDIA_API_KEY"] = "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"

from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

# COMMAND ----------

print(pprint)

# COMMAND ----------

from langchain_nvidia_ai_endpoints import ChatNVIDIA
# ChatNVIDIA.get_available_models()

# COMMAND ----------

## Useful utility method for printing intermediate states
from langchain_core.runnables import RunnableLambda
from functools import partial

def RPrint(preface="State: "):
    def print_and_return(x, preface=""):
        print(f"{preface}{x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def PPrint(preface="State: "):
    def print_and_return(x, preface=""):
        pprint(preface, x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

# COMMAND ----------

# MAGIC %md
# MAGIC ----
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ## **Part 1:** Keeping Variables Flowing
# MAGIC
# MAGIC In the previous examples, we were able to implement interesting logic in our standalone chains by **creating**, **mutating**, and **consuming** states. These states were passed around as dictionaries with descriptive keys and useful values, and the values would be used to supply follow-up routines with the info they need to operate!
# MAGIC
# MAGIC **Recall the zero-shot classification example from the last notebook:**

# COMMAND ----------

# MAGIC %%time
# MAGIC ## ^^ This notebook is timed, which will print out how long it all took
# MAGIC
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_nvidia_ai_endpoints import ChatNVIDIA
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import ChatPromptTemplate
# MAGIC from typing import List, Union
# MAGIC from operator import itemgetter
# MAGIC
# MAGIC ## Zero-shot classification prompt and chain w/ explicit few-shot prompting
# MAGIC sys_msg = (
# MAGIC     "Choose the most likely topic classification given the sentence as context."
# MAGIC     " Only one word, no explanation.\n[Options : {options}]"
# MAGIC )
# MAGIC
# MAGIC zsc_prompt = ChatPromptTemplate.from_template(
# MAGIC     f"{sys_msg}\n\n"
# MAGIC     "[[The sea is awesome]][/INST]boat</s><s>[INST]"
# MAGIC     "[[{input}]]"
# MAGIC )
# MAGIC
# MAGIC ## Define your simple instruct_model
# MAGIC instruct_chat = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.2")
# MAGIC instruct_llm = instruct_chat | StrOutputParser()
# MAGIC one_word_llm = instruct_chat.bind(stop=[" ", "\n"]) | StrOutputParser()
# MAGIC
# MAGIC zsc_chain = zsc_prompt | one_word_llm
# MAGIC
# MAGIC ## Function that just prints out the first word of the output. With early stopping bind
# MAGIC def zsc_call(input, options=["car", "boat", "airplane", "bike"]):
# MAGIC     return zsc_chain.invoke({"input" : input, "options" : options}).split()[0]
# MAGIC
# MAGIC print("-" * 80)
# MAGIC print(zsc_call("Should I take the next exit, or keep going to the next one?"))
# MAGIC
# MAGIC print("-" * 80)
# MAGIC print(zsc_call("I get seasick, so I think I'll pass on the trip"))
# MAGIC
# MAGIC print("-" * 80)
# MAGIC print(zsc_call("I'm scared of heights, so flying probably isn't for me"))

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC This chain makes several design decisions that make it very easy to use, key among them the following:
# MAGIC
# MAGIC **We want it to act like a function, so all we want it to do is generate the output and return it.**
# MAGIC
# MAGIC This makes the chain extremely natural for inclusion as a module in a larger chain system. For example, the following chain will take a string, extract the most likely topic, and then generate a new sentence based on the topic:
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %%time
# MAGIC ## ^^ This notebook is timed, which will print out how long it all took
# MAGIC gen_prompt = ChatPromptTemplate.from_template(
# MAGIC     "Make a new sentence about the the following topic: {topic}. Be creative!"
# MAGIC )
# MAGIC
# MAGIC gen_chain = gen_prompt | instruct_llm
# MAGIC
# MAGIC input_msg = "I get seasick, so I think I'll pass on the trip"
# MAGIC options = ["car", "boat", "airplane", "bike"]
# MAGIC
# MAGIC chain = (
# MAGIC     ## -> {"input", "options"}
# MAGIC     {'topic' : zsc_chain}
# MAGIC     | PPrint()
# MAGIC     ## -> {**, "topic"}
# MAGIC     | gen_chain
# MAGIC     ## -> string
# MAGIC )
# MAGIC
# MAGIC chain.invoke({"input" : input_msg, "options" : options})

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC However, it's a bit problematic when you want to keep the information flowing, since we lose the topic and input variables in generating our response. If we wanted to do something with both the output and the input, we'd need a way to make sure that both variables pass through.
# MAGIC
# MAGIC Lucky for us, we can use the mapping runnable (i.e. interpretted from a dictionary or using manual `RunnableMap`) to pass both of the variables through by assigning the output of our chain to just a single key and letting the other keys propagate as desired. Alternatively, we could also use `RunnableAssign` to merge the state-consuming chain's output with the input dictionary by default.
# MAGIC
# MAGIC With this technique, we can propagate whatever we want through our chain system:

# COMMAND ----------

# MAGIC %%time
# MAGIC ## ^^ This notebook is timed, which will print out how long it all took
# MAGIC
# MAGIC from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
# MAGIC from langchain.schema.runnable.passthrough import RunnableAssign
# MAGIC from functools import partial
# MAGIC
# MAGIC big_chain = (
# MAGIC     PPrint()
# MAGIC     ## Manual mapping. Can be useful sometimes and inside branch chains
# MAGIC     | {'input' : lambda d: d.get('input'), 'topic' : zsc_chain}
# MAGIC     | PPrint()
# MAGIC     ## RunnableAssign passing. Better for running state chains by default
# MAGIC     | RunnableAssign({'generation' : gen_chain})
# MAGIC     | PPrint()
# MAGIC     ## Using the input and generation together
# MAGIC     | RunnableAssign({'combination' : (
# MAGIC         ChatPromptTemplate.from_template(
# MAGIC             "Consider the following passages:"
# MAGIC             "\nP1: {input}"
# MAGIC             "\nP2: {generation}"
# MAGIC             "\n\nCombine the ideas from both sentences into one simple one."
# MAGIC         )
# MAGIC         | instruct_llm
# MAGIC     )})
# MAGIC )
# MAGIC
# MAGIC output = big_chain.invoke({
# MAGIC     "input" : "I get seasick, so I think I'll pass on the trip",
# MAGIC     "options" : ["car", "boat", "airplane", "bike", "unknown"]
# MAGIC })
# MAGIC pprint("Final Output: ", output)

# COMMAND ----------

# MAGIC %md
# MAGIC ----
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ## **Part 2:** Running State Chain
# MAGIC
# MAGIC The example above is just a toy example and, if anything, showcases the drawbacks of chaining many LLM calls together for internal under-the-hood reasoning. However, the ability to keep information flowing through a chain is invaluable for making complex chains that can accumulate useful state information or operate in a multi-pass capacity.
# MAGIC
# MAGIC Specifically, a very simple but effective chain is a **Running State Chain** which enforces the following properties:
# MAGIC - A **"running state"** is a dictionary that contains all of the variables that the system cares about.
# MAGIC - A **"branch"** is a chain that can pull in the running state and can degenerate it into a response.
# MAGIC - A **branch** can only be ran inside a **RunnableAssign** scope, and the branchs' inputs should come from the **running state**.

# COMMAND ----------

# MAGIC %md
# MAGIC > <img src="https://dli-lms.s3.amazonaws.com/assets/s-fx-15-v1/imgs/running_state_chain.png" width=1000px/>
# MAGIC <!-- > <img src="https://drive.google.com/uc?export=view&id=1Oo7AauYGj4dxepNReRG2JezmvQLyqXsN" width=1000px/> -->

# COMMAND ----------

# MAGIC %md
# MAGIC You can think of the running state chain abstraction as a functional variant of a Pythonic class with state variables (or attributes) and functions (or methods).
# MAGIC - The chain is like the abstract class that wraps all of the functionality.
# MAGIC - The running state are like the attributes (which should always be accessible).
# MAGIC - The branches are like the class methods (which can pick and choose which attributes to use).
# MAGIC - The `.invoke` or similar process is like the `__call__` method that runs through the branches in order.
# MAGIC
# MAGIC **By forcing this paradigm in your chains:**
# MAGIC - You can keep state variables propagating through your chain, allowing your internals to access whatever is necessary and accumulating state values for use later.
# MAGIC - You can also pass the outputs of your chain back through as your inputs, allowing a "while-loop"-style chain that keeps updating and building on your running state.
# MAGIC
# MAGIC The rest of this notebook will include two exercises that flesh out the running state chain abstraction for two additional use-cases: **Knowledge Bases** and **Database-Querying Chatbots**.

# COMMAND ----------

# MAGIC %md
# MAGIC ----
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ## **Part 3:** Implementing a Knowledge Base with Running State Chain
# MAGIC
# MAGIC After understanding the basic structure and principles of a Running State Chain, we can explore how this approach can be extended to manage more complex tasks, particularly in creating dynamic systems that evolve through interaction. This section will focus on implementing a **knowledge base** accumulated using **json-enabled slot filling**:
# MAGIC
# MAGIC - **Knowledge Base:** A store of information that's relevant for our LLM to keep track of.
# MAGIC - **JSON-Enabled Slot Filling:** The technique of asking an instruction-tuned model to output a json-style format (which can include a dictionary) with a selection of slots, relying on the LLM to fill these slots with useful and relevant information.

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC #### **Defining Our Knowledge Base**
# MAGIC
# MAGIC To build a responsive and intelligent system, we need a method that not only processes inputs but also retains and updates essential information through the flow of conversation. This is where the combination of LangChain and Pydantic becomes pivotal. [**Pydantic**](https://docs.pydantic.dev/latest/), a popular Python validation library, is instrumental in structuring and validating data models. As one of its features, Pydantic offers structured "model" classes that validate objects (data, classes, themselves, etc.) with simplified syntax and deep rabbitholes of customization options. This framework is used throughout LangChain and comes up as a necessary component for use cases that involve data coersion.
# MAGIC
# MAGIC One thing that a "model" is very good for is defining a class with expected arguments and some special ways to validate them! In this course, we won't focus too much on the validation scripts, but those interested can start by checking out the [**Pydantic Validator guide**](https://docs.pydantic.dev/1.10/usage/validators/) (though the topics do get pretty deep pretty fast). For our purposes, we can construct a `BaseModel` class and define some `Field` variables to create a structured **Knowledge Base** like so:

# COMMAND ----------

from pydantic import BaseModel, Field
from typing import Dict, Union, Optional

instruct_chat = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.2")

class KnowledgeBase(BaseModel):
    ## Fields of the BaseModel, which will be validated/assigned when the knowledge base is constructed
    topic: str = Field('general', description="Current conversation topic")
    user_preferences: Dict[str, Union[str, int]] = Field({}, description="User preferences and choices")
    session_notes: list = Field([], description="Notes on the ongoing session")
    unresolved_queries: list = Field([], description="Unresolved user queries")
    action_items: list = Field([], description="Actionable items identified during the conversation")

print(repr(KnowledgeBase(topic = "Travel")))

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC The true strength of this approach lies in the additional LLM-centric functionalities provided by LangChain which we can integrate for our use-cases. One such feature is the `PydanticOutputParser` which enhances the Pydantic objects with capabilities like automatic format instruction generation.

# COMMAND ----------

from langchain.output_parsers import PydanticOutputParser

instruct_string = PydanticOutputParser(pydantic_object=KnowledgeBase).get_format_instructions()
pprint(instruct_string)

# COMMAND ----------

# MAGIC %md
# MAGIC This functionality generates instructions for creating valid inputs to the Knowledge Base, which in turn helps the LLM by providing a concrete one-shot example of the desired output format.

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC #### **Runnable Extraction Module**
# MAGIC
# MAGIC Knowing that we have this Pydantic object which can be used to generate good LLM instructions, we can make a Runnable that wraps the functionality of our Pydantic class and streamlines the prompting, generating, and updating of the knowledge base:

# COMMAND ----------

################################################################################
## Definition of RExtract
def RExtract(pydantic_class, llm, prompt):
    '''
    Runnable Extraction module
    Returns a knowledge dictionary populated by slot-filling extraction
    '''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions' : lambda x: parser.get_format_instructions()})
    def preparse(string):
        if '{' not in string: string = '{' + string
        if '}' not in string: string = string + '}'
        string = (string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        )
        # print(string)  ## Good for diagnostics
        return string
    return instruct_merge | prompt | llm | preparse | parser

################################################################################
## Practical Use of RExtract

parser_prompt = ChatPromptTemplate.from_template(
    "Update the knowledge base: {format_instructions}. Only use information from the input."
    "\n\nNEW MESSAGE: {input}"
)

extractor = RExtract(KnowledgeBase, instruct_llm, parser_prompt)

knowledge = extractor.invoke({'input' : "I love flowers so much! The orchids are amazing! Can you buy me some?"})
pprint(knowledge)

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC Do keep in mind that this process can fail due to the fuzzy nature of LLM prediction, especially with models that are not optimized for instruction-following! For this process, it's important to have a strong instruction-following LLM with extra checks and graceful failure routines. 

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC #### **Dynamic Knowledge Base Updates**
# MAGIC
# MAGIC Finally, we can create a system that continually updates the Knowledge Base throughout the conversation. This is done by feeding the current state of the Knowledge Base, along with new user inputs, back into the system for ongoing updates.
# MAGIC
# MAGIC The following is an example system that shows off both the formulation's power of filling details as well as the limitations of assuming that filling performance will be as good as general response performance:

# COMMAND ----------

class KnowledgeBase(BaseModel):
    firstname: str = Field('unknown', description="Chatting user's first name, unknown if unknown")
    lastname: str = Field('unknown', description="Chatting user's last name, unknown if unknown")
    location: str = Field('unknown', description="Where the user is located")
    summary: str = Field('unknown', description="Running summary of conversation. Update this with new input")
    response: str = Field('unknown', description="An ideal response to the user based on their new message")


parser_prompt = ChatPromptTemplate.from_template(
    "You are chatting with a user. The user just responded ('input'). Please update the knowledge base."
    " Record your response in the 'response' tag to continue the conversation."
    " Do not hallucinate any details, and make sure the knowledge base is not redundant."
    " Update the entries frequently to adapt to the conversation flow."
    "\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nNEW MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE:"
)

## Switch to a more powerful base model
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()

extractor = RExtract(KnowledgeBase, instruct_llm, parser_prompt)
info_update = RunnableAssign({'know_base' : extractor})

## Initialize the knowledge base and see what you get
state = {'know_base' : KnowledgeBase()}
state['input'] = "My name is Carmen Sandiego! Guess where I am! Hint: It's somewhere in the United States."
state = info_update.invoke(state)
pprint(state)

# COMMAND ----------

state['input'] = "I'm in a place considered the birthplace of Jazz."
state = info_update.invoke(state)
pprint(state)

# COMMAND ----------

state['input'] = "Yeah, I'm in New Orleans... How did you know?"
state = info_update.invoke(state)
pprint(state)

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC This example demonstrates how a running state chain can be effectively utilized to manage a conversation with evolving context and requirements, making it a powerful tool for developing sophisticated interactive systems.
# MAGIC
# MAGIC The next sections of this notebook will expand on these concepts by exploring two specific applications: **Document Knowledge Bases** and **Database-Querying Chatbots**.

# COMMAND ----------

# MAGIC %md
# MAGIC ----
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ## **Part 4: [Exercise]** Airline Customer Service Bot
# MAGIC
# MAGIC In this exercise, we can expand on the tools we've learned about to implement a simple but effective dialog manager chatbot. For this exercise, we will make an airline support bot that wants to help a client find out about their flight!
# MAGIC
# MAGIC Let's create a simple database-like interface to get some customer information from a dictionary!

# COMMAND ----------

#######################################################################################
## Function that can be queried for information. Implementation details not important
def get_flight_info(d: dict) -> str:
    """
    Example of a retrieval function which takes a dictionary as key. Resembles SQL DB Query
    """
    req_keys = ['first_name', 'last_name', 'confirmation']
    assert all((key in d) for key in req_keys), f"Expected dictionary with keys {req_keys}, got {d}"

    ## Static dataset. get_key and get_val can be used to work with it, and db is your variable
    keys = req_keys + ["departure", "destination", "departure_time", "arrival_time", "flight_day"]
    values = [
        ["Jane", "Doe", 12345, "San Jose", "New Orleans", "12:30 PM", "9:30 PM", "tomorrow"],
        ["John", "Smith", 54321, "New York", "Los Angeles", "8:00 AM", "11:00 AM", "Sunday"],
        ["Alice", "Johnson", 98765, "Chicago", "Miami", "7:00 PM", "11:00 PM", "next week"],
        ["Bob", "Brown", 56789, "Dallas", "Seattle", "1:00 PM", "4:00 PM", "yesterday"],
    ]
    get_key = lambda d: "|".join([d['first_name'], d['last_name'], str(d['confirmation'])])
    get_val = lambda l: {k:v for k,v in zip(keys, l)}
    db = {get_key(get_val(entry)) : get_val(entry) for entry in values}

    # Search for the matching entry
    data = db.get(get_key(d))
    if not data:
        return (
            f"Based on {req_keys} = {get_key(d)}) from your knowledge base, no info on the user flight was found."
            " This process happens every time new info is learned. If it's important, ask them to confirm this info."
        )
    return (
        f"{data['first_name']} {data['last_name']}'s flight from {data['departure']} to {data['destination']}"
        f" departs at {data['departure_time']} {data['flight_day']} and lands at {data['arrival_time']}."
    )

#######################################################################################
## Usage example. Actually important

print(get_flight_info({"first_name" : "Jane", "last_name" : "Doe", "confirmation" : 12345}))

# COMMAND ----------

print(get_flight_info({"first_name" : "Alice", "last_name" : "Johnson", "confirmation" : 98765}))

# COMMAND ----------

print(get_flight_info({"first_name" : "Bob", "last_name" : "Brown", "confirmation" : 27494}))

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC This is a really good interface to bring up because it can reasonably serve two purposes:
# MAGIC - It can be used to provide up-to-date information from an external environment (a database) regarding a user's situation.
# MAGIC - It can also be used as a hard gating mechanism to prevent unauthorized disclosure of sensitive information (since that would be very bad).
# MAGIC
# MAGIC If our network had access to this kind of interface, it would be able to query for and retrieve this information on a user's behalf! For example:

# COMMAND ----------

external_prompt = ChatPromptTemplate.from_template(
    "You are a SkyFlow chatbot, and you are helping a customer with their issue."
    " Please help them with their question, remembering that your job is to represent SkyFlow airlines."
    " Assume SkyFlow uses industry-average practices regarding arrival times, operations, etc."
    " (This is a trade secret. Do not disclose)."  ## soft reinforcement
    " Please keep your discussion short and sweet if possible. Avoid saying hello unless necessary."
    " The following is some context that may be useful in answering the question."
    "\n\nContext: {context}"
    "\n\nUser: {input}"
)

basic_chain = external_prompt | instruct_llm

basic_chain.invoke({
    'input' : 'Can you please tell me when I need to get to the airport?',
    'context' : get_flight_info({"first_name" : "Jane", "last_name" : "Doe", "confirmation" : 12345}),
})

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC This is interesting enough, but how do we actually get this system working in the wild? It turns out that we can use the KnowledgeBase formulation from above to supply this kind of information like so:

# COMMAND ----------

from pydantic import BaseModel, Field
from typing import Dict, Union

class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="Chatting user's first name, `unknown` if unknown")
    last_name: str = Field('unknown', description="Chatting user's last name, `unknown` if unknown")
    confirmation: int = Field(-1, description="Flight Confirmation Number, `-1` if unknown")
    discussion_summary: str = Field("", description="Summary of discussion so far, including locations, issues, etc.")
    open_problems: list = Field([], description="Topics that have not been resolved yet")
    current_goals: list = Field([], description="Current goal for the agent to address")

def get_key_fn(base: BaseModel) -> dict:
    '''Given a dictionary with a knowledge base, return a key for get_flight_info'''
    return {  ## More automatic options possible, but this is more explicit
        'first_name' : base.first_name,
        'last_name' : base.last_name,
        'confirmation' : base.confirmation,
    }

know_base = KnowledgeBase(first_name = "Jane", last_name = "Doe", confirmation = 12345)

# get_flight_info(get_key_fn(know_base))

get_key = RunnableLambda(get_key_fn)
(get_key | get_flight_info).invoke(know_base)

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC ### **Objective:**
# MAGIC
# MAGIC You want a user to be able to invoke the following function call organically as part of a dialog exchange:
# MAGIC
# MAGIC ```python
# MAGIC get_flight_info({"first_name" : "Jane", "last_name" : "Doe", "confirmation" : 12345}) ->
# MAGIC     "Jane Doe's flight from San Jose to New Orleans departs at 12:30 PM tomorrow and lands at 9:30 PM."
# MAGIC ```
# MAGIC
# MAGIC `RExtract` is provided such that the following knowledge base syntax can be used:
# MAGIC ```python
# MAGIC known_info = KnowledgeBase()
# MAGIC extractor = RExtract(KnowledgeBase, InstructLLM(), parser_prompt)
# MAGIC results = extractor.invoke({'info_base' : known_info, 'input' : 'My message'})
# MAGIC known_info = results['info_base']
# MAGIC ```
# MAGIC
# MAGIC **Design a chatbot that implements the following features:**
# MAGIC - The bot should start off by making small-talk, possibly helping the user with non-sensitive queries which don't require any private info access.
# MAGIC - When the user starts to ask about things that are database-walled (both practically and legally), tell the user that they need to provide the relevant information.
# MAGIC - When the retrieval is successful, the agent will be able to talk about the database-walled information.
# MAGIC
# MAGIC **This can be done with a variety of techniques, including the following:**
# MAGIC - **Prompt Engineering and Context Parsing**, where the overall chat prompt stays roughly the same but the context is manipulated to to change agent behavior. For example, a failed db retrieval could be changed into an injection of natural-language instructions for how to resolve the problem such as *`"Information could not be retrieved with keys {...}. Please ask the user for clarification or help them with known information."`*
# MAGIC - **"Prompt Passing,"** where the active prompts are passed around as state variables and can be overridden by monitoring chains.
# MAGIC - **Branching chains** such as [**`RunnableBranch`**](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.branch.RunnableBranch.html) or more custom solutions that implement an conditional routing mechanism.
# MAGIC     - In the case of [`RunnableBranch`](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.branch.RunnableBranch.html), a `switch` syntax of the style:
# MAGIC         ```python
# MAGIC         from langchain.schema.runnable import RunnableBranch
# MAGIC         RunnableBranch(
# MAGIC             ((lambda x: 1 in x), RPrint("Has 1 (didn't check 2): ")),
# MAGIC             ((lambda x: 2 in x), RPrint("Has 2 (not 1 though): ")),
# MAGIC             RPrint("Has neither 1 not 2: ")
# MAGIC         ).invoke([2, 1, 3]);  ## -> Has 1 (didn't check 2): [2, 1, 3]
# MAGIC         ```
# MAGIC
# MAGIC Some prompts and a gradio loop are provided that might help with the effort, but the agent will currently just hallucinate! Please implement the internal chain to try and retrieve the relevant information. Before trying to implement, look over the default behavior of the model and note how it might hallucinate or forget things.

# COMMAND ----------

# MAGIC %pip install gradio

# COMMAND ----------

from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableMap,       ## Wrap an implicit "dictionary" runnable
    RunnablePassthrough,
)
from langchain.schema.runnable.passthrough import RunnableAssign

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, ChatMessage, AIMessage
from typing import Iterable
import gradio as gr

external_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a chatbot for SkyFlow Airlines, and you are helping a customer with their issue."
        " Please chat with them! Stay concise and clear!"
        " Your running knowledge base is: {know_base}."
        " This is for you only; Do not mention it!"
        " \nUsing that, we retrieved the following: {context}\n"
        " If they provide info and the retrieval fails, ask to confirm their first/last name and confirmation."
        " Do not ask them any other personal info."
        " If it's not important to know about their flight, do not ask."
        " The checking happens automatically; you cannot check manually."
    )),
    ("assistant", "{output}"),
    ("user", "{input}"),
])

##########################################################################
## Knowledge Base Things

class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="Chatting user's first name, `unknown` if unknown")
    last_name: str = Field('unknown', description="Chatting user's last name, `unknown` if unknown")
    confirmation: Optional[int] = Field(None, description="Flight Confirmation Number, `-1` if unknown")
    discussion_summary: str = Field("", description="Summary of discussion so far, including locations, issues, etc.")
    open_problems: str = Field("", description="Topics that have not been resolved yet")
    current_goals: str = Field("", description="Current goal for the agent to address")

parser_prompt = ChatPromptTemplate.from_template(
    "You are a chat assistant representing the airline SkyFlow, and are trying to track info about the conversation."
    " You have just received a message from the user. Please fill in the schema based on the chat."
    "\n\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nASSISTANT RESPONSE: {output}"
    "\n\nUSER MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE: "
)

## Your goal is to invoke the following through natural conversation
# get_flight_info({"first_name" : "Jane", "last_name" : "Doe", "confirmation" : 12345}) ->
#     "Jane Doe's flight from San Jose to New Orleans departs at 12:30 PM tomorrow and lands at 9:30 PM."

chat_llm = ChatNVIDIA(model="meta-llama/Llama-2-70b-chat-hf") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct") | StrOutputParser()

external_chain = external_prompt | chat_llm

#####################################################################################
## START TODO: Define the extractor and internal chain to satisfy the objective
from langchain.output_parsers import PydanticOutputParser

# ✅ Parser to convert JSON output into a KnowledgeBase object
kb_parser = PydanticOutputParser(pydantic_object=KnowledgeBase)

# ✅ Generate proper JSON formatting instructions for the LLM
format_instructions = kb_parser.get_format_instructions()

# ✅ Chain that updates the KnowledgeBase based on the assistant response + user message
parser_chain = (
    parser_prompt |
    instruct_llm |    # LLM infers the updated knowledge state
    kb_parser         # Parses JSON into KnowledgeBase object
)

# ✅ KnowledgeBase getter function - extracts & updates knowledge
def knowbase_getter(state):
    return parser_chain.invoke({
        "format_instructions": format_instructions,
        "know_base": state.get("know_base", KnowledgeBase()).json(),
        "output": state.get("output", ""),
        "input": state.get("input", "")
    })

# ✅ Fake DB lookup for demo
def get_flight_info(query: dict) -> str:
    if query.get("confirmation") == 12345:
        return "Jane Doe's flight from San Jose to New Orleans departs at 12:30 PM tomorrow and lands at 9:30 PM."
    elif query.get("confirmation") == 98765:
        return "John Smith’s flight from New York to Los Angeles departs at 5:00 PM today and lands at 8:30 PM."
    else:
        return "No matching flight found in our database."

# ✅ Database getter - uses updated KnowledgeBase to retrieve flight info
def database_getter(state):
    kb: KnowledgeBase = state.get("know_base", KnowledgeBase())
    
    # Only query DB if we have a valid confirmation number
    if kb.confirmation and kb.confirmation != -1:
        return get_flight_info({
            "first_name": kb.first_name,
            "last_name": kb.last_name,
            "confirmation": kb.confirmation
        })
    else:
        return "No flight details available yet."

# ✅ Combine into internal_chain
internal_chain = RunnableBranch({
    'know_base': knowbase_getter,
    'context': database_getter
})

## TODO: Make a chain that will populate your knowledge base based on provided context
# knowbase_getter = lambda x: KnowledgeBase()

## TODO: Make a chain to pull d["know_base"] and outputs a retrieval from db
# database_getter = lambda x: "Not implemented"

## These components integrate to make your internal chain
internal_chain = RunnableBranch({
    'know_base' : knowbase_getter,
    'context' : database_getter
})

## END TODO
#####################################################################################

state = {'know_base' : KnowledgeBase()}

def chat_gen(message, history=[], return_buffer=True):

    ## Pulling in, updating, and printing the state
    global state
    state['input'] = message
    state['history'] = history
    state['output'] = "" if not history else history[-1][1]

    ## Generating the new state from the internal chain
    state = internal_chain.invoke(state)
    print("State after chain run:")
    pprint({k:v for k,v in state.items() if k != "history"})
    
    ## Streaming the results
    buffer = ""
    for token in external_chain.stream(state):
        buffer += token
        yield buffer if return_buffer else token

def queue_fake_streaming_gradio(chat_stream, history = [], max_questions=8):

    ## Mimic of the gradio initialization routine, where a set of starter messages can be printed off
    for human_msg, agent_msg in history:
        if human_msg: print("\n[ Human ]:", human_msg)
        if agent_msg: print("\n[ Agent ]:", agent_msg)

    ## Mimic of the gradio loop with an initial message from the agent.
    for _ in range(max_questions):
        message = input("\n[ Human ]: ")
        print("\n[ Agent ]: ")
        history_entry = [message, ""]
        for token in chat_stream(message, history, return_buffer=False):
            print(token, end='')
            history_entry[1] += token
        history += [history_entry]
        print("\n")

## history is of format [[User response 0, Bot response 0], ...]
chat_history = [[None, "Hello! I'm your SkyFlow agent! How can I help you?"]]

## Simulating the queueing of a streaming gradio interface, using python input
queue_fake_streaming_gradio(
    chat_stream = chat_gen,
    history = chat_history
)

# COMMAND ----------

from langchain.output_parsers import PydanticOutputParser

# ✅ Parser to convert JSON output into a KnowledgeBase object
kb_parser = PydanticOutputParser(pydantic_object=KnowledgeBase)

# ✅ Generate proper JSON formatting instructions for the LLM
format_instructions = kb_parser.get_format_instructions()

# ✅ Chain that updates the KnowledgeBase based on the assistant response + user message
parser_chain = (
    parser_prompt |
    instruct_llm |    # LLM infers the updated knowledge state
    kb_parser         # Parses JSON into KnowledgeBase object
)

# ✅ KnowledgeBase getter function - extracts & updates knowledge
def knowbase_getter(state):
    return parser_chain.invoke({
        "format_instructions": format_instructions,
        "know_base": state.get("know_base", KnowledgeBase()).json(),
        "output": state.get("output", ""),
        "input": state.get("input", "")
    })

# ✅ Fake DB lookup for demo
def get_flight_info(query: dict) -> str:
    if query.get("confirmation") == 12345:
        return "Jane Doe's flight from San Jose to New Orleans departs at 12:30 PM tomorrow and lands at 9:30 PM."
    elif query.get("confirmation") == 98765:
        return "John Smith’s flight from New York to Los Angeles departs at 5:00 PM today and lands at 8:30 PM."
    else:
        return "No matching flight found in our database."

# ✅ Database getter - uses updated KnowledgeBase to retrieve flight info
def database_getter(state):
    kb: KnowledgeBase = state.get("know_base", KnowledgeBase())
    
    # Only query DB if we have a valid confirmation number
    if kb.confirmation and kb.confirmation != -1:
        return get_flight_info({
            "first_name": kb.first_name,
            "last_name": kb.last_name,
            "confirmation": kb.confirmation
        })
    else:
        return "No flight details available yet."

# ✅ Combine into internal_chain
internal_chain = (
    RunnableAssign({'know_base': knowbase_getter})
    | RunnableAssign({'context': database_getter})
)


# COMMAND ----------

# DBTITLE 1,1

#state = {'know_base' : KnowledgeBase()}

#chatbot = gr.Chatbot(value=[[None, "Hello! I'm your SkyFlow agent! How can I help you?"]])
#demo = gr.ChatInterface(chat_gen, chatbot=chatbot).queue().launch(debug=True, share=True)

# COMMAND ----------

# MAGIC %md
# MAGIC <br>
# MAGIC
# MAGIC ----
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC **NOTE:**
# MAGIC - You may need to explicitly hit the STOP button and try to relaunch your gradio interface if it hangs up after an exception. This is a known Jupyter Notebook environment issue which should not be experienced in dedicated Gradio-running files.
# MAGIC - **Your chat directive is duplicated here for quick access:**
# MAGIC ```python
# MAGIC ## Your goal is to invoke the following through natural conversation
# MAGIC get_flight_info({
# MAGIC     "first_name" : "Jane",
# MAGIC     "last_name" : "Doe",
# MAGIC     "confirmation" : 12345,
# MAGIC }) -> "Jane Doe's flight from San Jose to New Orleans departs at 12:30 PM tomorrow and lands at 9:30 PM."
# MAGIC ```
# MAGIC - **To confirm that your system works, you could try the following dialog or something similar:**
# MAGIC ```
# MAGIC > How's it going?
# MAGIC > Can you tell me a bit about skyflow?
# MAGIC > Can you tell me about my flight?
# MAGIC > My name is Jane Doe and my flight confirmation is 12345
# MAGIC > Can you tell me when I should get to my flight?
# MAGIC ```
# MAGIC - **Solutions To Exercises Can Be Found In The Solutions Directory.** This is the first exercise with a noted solution, and additional exercises from the future notebooks will be found there.

# COMMAND ----------

# MAGIC %md
# MAGIC -----
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ## **Part 5:** Wrap-Up
# MAGIC
# MAGIC The goal of this notebook was to introduce some more advanced LangChain material revolving around the use of knowledge bases and running state chains! The exercise here was pretty involved, so congrats on finishing it!
# MAGIC
# MAGIC ### <font color="#76b900">**Great Job!**</font>
# MAGIC ----