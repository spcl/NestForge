## When something fails

- A refused move changes nothing and names the reason: a label a move retired (read the tree again and use
  its labels), or a move that is not legal. Do not resend it unchanged.
- A move that makes the program slower or wrong: apply its inverse when `list_moves` offers one, else go back to
  the last library you measured.
- A refusal (4xx) grades nothing and names what was expected next to what arrived. `judge_fault: true`, or a 500
  saying `score failed`, means the judge failed and your moves did not: retry once.
- Your moves are never lost: not by a timeout, not by a 503 and not by a judge fault. A call that ran out of time keeps
  what it built and measured, so calling it again goes on from there. `applied_moves` lists what the program holds.
