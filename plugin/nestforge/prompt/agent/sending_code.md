## Sending code

`finalize` writes the program as a library in the bench C ABI to the shared folder and answers with `library` and
`workspace_bytes`. Send both unchanged: `score` and `submit` with `library=<library>` and
`workspace_bytes=<workspace_bytes>`. Call `finalize` again after every change you keep; a library from before a
change is the program before it.
