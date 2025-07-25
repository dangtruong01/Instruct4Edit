### Prompt Used for Instruction Generation

{few_shot_examples}
You are a frontend UI expert. Given this HTML page, write 5 different, specific, visible design edit instructions as if you were giving them to a designer or developer in natural language.

Instructions should be HUMAN-LIKE and must NOT mention or reference any code, class names, ids, or HTML tags directly.

Focus on describing the desired visual change, layout, or style in plain English.

Examples of human-like instructions:
- Make the navigation bar background a darker color.
- Add more space between the sections.
- Center the main heading on the page.
- Make all buttons rounded.
- Hide the sidebar on mobile screens.

Bad example (do NOT do this): 'Add border-radius: 5px to .container'

HTML:
{html_code}

Respond ONLY with the 5 instructions, numbered 1 to 5, and nothing else. Do not add explanations or code references.

Instructions:
1.