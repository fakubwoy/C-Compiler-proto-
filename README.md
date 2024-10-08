# C-Compiler-proto

Prototype C compiler written in C++

## Phases of Compilation

1. **Preprocessing**
   - Remove comments and header files from the source code.
   - Output: Cleaned source code without comments and headers.

2. **Lexical Analysis**
   - Split the contents of the file into tokens.
   - Output: List of tokens extracted from the source code.

3. **Syntax Analysis**
   - Construct the Abstract Syntax Tree (AST) from the tokens.
   - Output: AST representation of the code.

4. **Semantic Analysis**
   - Parse and validate the AST to ensure correct semantics.
   - Output: Annotated AST with semantic information.

5. **Intermediate Code Generation**
   - Generate the three-address code (TAC) from the AST.
   - Output: TAC representation of the code.

6. **Code Optimization**
   - Optimize the intermediate code for performance.
   - Output: Optimized TAC.

7. **Convert to Assembly**
   - Convert the intermediate code to assembly language (x86).
   - Output: x86 assembly code.

# ToDo:
- Basic arithmetic has been handled
- Variables have been handled
- Basic boolean comparison left (if-else)
- Function definition left
