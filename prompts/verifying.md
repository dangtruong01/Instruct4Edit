### Prompt Used for Verifying

You are a highly cautious visual UI/UX reviewer. I will show you two images: one "before" and one "after" a design change was instructed.
- The FIRST image is the ORIGINAL (before the design change).
- The SECOND image is the MODIFIED (after the design change was applied).

Your task:
Compare the two images and verify whether the following instruction was fully and clearly implemented in the modified image.
If you're unsure, highlight what might be missing or only partially done.

Instruction:
{instruction}

Step-by-step justification:
1. What is the intended visual change?
2. Is this change clearly visible in the after image compared to the before image?
3. Are there any doubts, ambiguities, or missing elements?

Final verdict (choose only one):
- ✅ Fully Applied
- ❌ Not Applied

Now compare the two images and respond accordingly.
Only take VISUAL changes into consideration.