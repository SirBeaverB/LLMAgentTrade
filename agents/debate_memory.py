from typing import Dict, Any, List
from agents import BaseAgent

class MemorySummaryAgent(BaseAgent):
    """
    A MemorySummaryAgent is responsible for taking a set of arguments (e.g., from a debate round),
    summarizing them into one or two concise sentences, and then storing this summary into the
    appropriate memory structure (e.g., mid-term or short-term memory).

    Usage:
    - Initialize with a config.
    - Call `summarize_speeches` with a list of argument texts.
    - The agent will produce a short summary and return it. You can then integrate this summary
      into your DebateAgent's mid-term or short-term memory.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "MemorySummaryAgent analyze not implemented."}

    def summarize_speeches(self, texts: List[str], context: str = "") -> str:
        """
        Summarize a list of argument texts into one or two sentences.
        
        Args:
            texts: A list of strings, each representing an analyst's argument.
            context: Optional additional context to inform the summarization.

        Returns:
            A short summary (one or two sentences) capturing the essence of the provided arguments.
        """
        # Combine texts into a single input, possibly limit length if needed
        combined_text = " ".join(t.strip() for t in texts)
        role = "You are a highly concise summarizer."
        content = f"""
        Below are several analysts' statements:
        {combined_text}

        {context}

        Please summarize the above arguments into one or two concise sentences
        that capture the main essence of their content.
        Keep it very brief and to-the-point.
        """

        summary = self._create_prompt(role, content)
        # The returned summary might be more than two sentences; if so, we can post-process.
        final_summary = self._enforce_length(summary)
        return final_summary

    def _enforce_length(self, text: str) -> str:
        """
        Ensure the summary is at most two sentences.
        If there's more than two sentences, keep only the first two.
        """
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 2:
            sentences = sentences[:2]
        final_text = ". ".join(sentences)
        if not final_text.endswith('.'):
            final_text += '.'
        return final_text

    def add_to_mid_term_memory(self, memory_list: List[str], summary: str):
        """
        Add the given summary to the mid-term memory list.
        """
        memory_list.append(summary)

    def add_to_short_term_memory(self, memory_list: List[str], summary: str):
        """
        Add the given summary to the short-term memory list.
        """
        memory_list.append(summary)