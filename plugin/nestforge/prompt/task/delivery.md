## Delivery: the library NestForge assembles
You shape the program with the NestForge tools, and `finalize` turns it into a prebuilt shared library `{{ lib_name }}`
exporting `{{ symbol }}` in the shared folder `{{ shared_dir }}`.

`finalize` answers with `library` and `workspace_bytes`. Pass both unchanged to `score` while you iterate and to
`submit` at the end: set the response `library` field to that path and `workspace_bytes` to that value. Call
`finalize` again after every change you keep; a library from before a change is the program before it.
