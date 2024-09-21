#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <unordered_set>
enum class TokenType {
    INTEGER,//0
    IDENTIFIER,//1
    PLUS,//2
    MINUS,//3
    MULTIPLY,//4
    DIVIDE,//5
    LPAREN,//6
    RPAREN,//7
    SEMICOLON,//8
    ASSIGN,//9
    EOF_TOKEN//10
};

struct Token {
    TokenType type;
    std::string value;
    Token(TokenType t, const std::string& v) : type(t), value(v) {}
};

enum class NodeType {
    BINARY_OP,
    INTEGER,
    VARIABLE,
    ASSIGNMENT,
    RETURN
};

class ASTNode {
public:
    NodeType type;
    explicit ASTNode(NodeType t) : type(t) {}
    virtual ~ASTNode() = default;
    virtual std::string toString() const = 0;
};

class IntegerNode : public ASTNode {
public:
    int value;
    IntegerNode(int val) : ASTNode(NodeType::INTEGER), value(val) {}
    std::string toString() const override {
        return "Integer: " + std::to_string(value);
    }
};

class VariableNode : public ASTNode {
public:
    std::string name;
    VariableNode(const std::string& n) : ASTNode(NodeType::VARIABLE), name(n) {}
    std::string toString() const override {
        return "Variable: " + name;
    }
};

class BinaryOpNode : public ASTNode {
public:
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;
    TokenType op;
    BinaryOpNode(std::unique_ptr<ASTNode> l, std::unique_ptr<ASTNode> r, TokenType o)
        : ASTNode(NodeType::BINARY_OP), left(std::move(l)), right(std::move(r)), op(o) {}
    std::string toString() const override {
        std::string opStr;
        switch (op) {
            case TokenType::PLUS: opStr = "+"; break;
            case TokenType::MINUS: opStr = "-"; break;
            case TokenType::MULTIPLY: opStr = "*"; break;
            case TokenType::DIVIDE: opStr = "/"; break;
            default: opStr = "Unknown";
        }
        return "BinaryOp: " + opStr + "\n  Left: " + left->toString() + "\n  Right: " + right->toString();
    }
};

class AssignmentNode : public ASTNode {
public:
    std::string variable;
    std::unique_ptr<ASTNode> expression;
    AssignmentNode(const std::string& var, std::unique_ptr<ASTNode> expr)
        : ASTNode(NodeType::ASSIGNMENT), variable(var), expression(std::move(expr)) {}
    std::string toString() const override {
        return "Assignment:\n  Variable: " + variable + "\n  Expression: " + expression->toString();
    }
};

class ReturnNode : public ASTNode {
public:
    std::unique_ptr<ASTNode> expression;
    ReturnNode(std::unique_ptr<ASTNode> expr) : ASTNode(NodeType::RETURN), expression(std::move(expr)) {}
    std::string toString() const override {
        return "Return:\n  Expression: " + expression->toString();
    }
};

class Lexer {
private:
    std::string input;
    size_t position;

public:
    explicit Lexer(const std::string& source) : input(source), position(0) {}

    Token getNextToken() {
        while (position < input.length() && std::isspace(input[position])) {
            position++;
        }

        if (position >= input.length()) {
            return Token(TokenType::EOF_TOKEN, "");
        }

        if (std::isalpha(input[position])) {
            std::string identifier;
            while (position < input.length() && (std::isalnum(input[position]) || input[position] == '_')) {
                identifier += input[position++];
            }
            return Token(TokenType::IDENTIFIER, identifier);
        }

        if (std::isdigit(input[position])) {
            std::string num;
            while (position < input.length() && std::isdigit(input[position])) {
                num += input[position++];
            }
            return Token(TokenType::INTEGER, num);
        }

        switch (input[position]) {
            case '+': position++; return Token(TokenType::PLUS, "+");
            case '-': position++; return Token(TokenType::MINUS, "-");
            case '*': position++; return Token(TokenType::MULTIPLY, "*");
            case '/': position++; return Token(TokenType::DIVIDE, "/");
            case '(': position++; return Token(TokenType::LPAREN, "(");
            case ')': position++; return Token(TokenType::RPAREN, ")");
            case ';': position++; return Token(TokenType::SEMICOLON, ";");
            case '=': position++; return Token(TokenType::ASSIGN, "=");
            default:
                throw std::runtime_error("Invalid character encountered: " + std::string(1, input[position]));
        }
    }
};

class Parser {
private:
    Lexer lexer;
    Token currentToken;

    void eat(TokenType tokenType) {
        if (currentToken.type == tokenType) {
            currentToken = lexer.getNextToken();
        } else {
            throw std::runtime_error("Unexpected token: expected " + std::to_string(static_cast<int>(tokenType)) +
                                     ", got " + std::to_string(static_cast<int>(currentToken.type)));
        }
    }

    std::unique_ptr<ASTNode> factor() {
        if (currentToken.type == TokenType::INTEGER) {
            int value = std::stoi(currentToken.value);
            eat(TokenType::INTEGER);
            return std::make_unique<IntegerNode>(value);
        } else if (currentToken.type == TokenType::IDENTIFIER) {
            std::string name = currentToken.value;
            eat(TokenType::IDENTIFIER);
            return std::make_unique<VariableNode>(name);
        } else if (currentToken.type == TokenType::LPAREN) {
            eat(TokenType::LPAREN);
            auto node = expr();
            eat(TokenType::RPAREN);
            return node;
        }
        throw std::runtime_error("Invalid factor");
    }

    std::unique_ptr<ASTNode> term() {
        auto node = factor();

        while (currentToken.type == TokenType::MULTIPLY || currentToken.type == TokenType::DIVIDE) {
            Token op = currentToken;
            if (op.type == TokenType::MULTIPLY) {
                eat(TokenType::MULTIPLY);
            } else if (op.type == TokenType::DIVIDE) {
                eat(TokenType::DIVIDE);
            }

            node = std::make_unique<BinaryOpNode>(std::move(node), factor(), op.type);
        }

        return node;
    }

    std::unique_ptr<ASTNode> expr() {
        auto node = term();

        while (currentToken.type == TokenType::PLUS || currentToken.type == TokenType::MINUS) {
            Token op = currentToken;
            if (op.type == TokenType::PLUS) {
                eat(TokenType::PLUS);
            } else if (op.type == TokenType::MINUS) {
                eat(TokenType::MINUS);
            }

            node = std::make_unique<BinaryOpNode>(std::move(node), term(), op.type);
        }

        return node;
    }

    std::unique_ptr<ASTNode> assignment() {
        if (currentToken.type == TokenType::IDENTIFIER) {
            std::string varName = currentToken.value;
            eat(TokenType::IDENTIFIER);
            eat(TokenType::ASSIGN);
            auto exprNode = expr();
            return std::make_unique<AssignmentNode>(varName, std::move(exprNode));
        }
        return expr();
    }

    std::unique_ptr<ASTNode> statement() {
        if (currentToken.type == TokenType::IDENTIFIER && currentToken.value == "return") {
            eat(TokenType::IDENTIFIER);
            auto returnExpr = expr();
            eat(TokenType::SEMICOLON);
            return std::make_unique<ReturnNode>(std::move(returnExpr));
        } else {
            auto node = assignment();
            eat(TokenType::SEMICOLON);
            return node;
        }
    }

public:
    explicit Parser(const std::string& input) : lexer(input), currentToken(TokenType::EOF_TOKEN, "") {
        currentToken = lexer.getNextToken();
    }

    std::vector<std::unique_ptr<ASTNode>> parse() {
        std::vector<std::unique_ptr<ASTNode>> statements;
        while (currentToken.type != TokenType::EOF_TOKEN) {
            statements.push_back(statement());
        }
        return statements;
    }
};

class SemanticAnalyzer {
private:
    std::unordered_map<std::string, bool> variables;

public:
    void analyze(const ASTNode* node) {
        if (node->type == NodeType::INTEGER) {

        } else if (node->type == NodeType::VARIABLE) {
            const auto* varNode = static_cast<const VariableNode*>(node);
            if (variables.find(varNode->name) == variables.end()) {
                throw std::runtime_error("Undefined variable: " + varNode->name);
            }
        } else if (node->type == NodeType::BINARY_OP) {
            const auto* binOp = static_cast<const BinaryOpNode*>(node);
            analyze(binOp->left.get());
            analyze(binOp->right.get());
        } else if (node->type == NodeType::ASSIGNMENT) {
            const auto* assignNode = static_cast<const AssignmentNode*>(node);
            variables[assignNode->variable] = true;
            analyze(assignNode->expression.get());
        } else if (node->type == NodeType::RETURN) {
            const auto* returnNode = static_cast<const ReturnNode*>(node);
            analyze(returnNode->expression.get());
        }
    }
};

class IntermediateCodeGenerator {
private:
    int tempCounter = 0;
    std::string generateTemp() {
        return "t" + std::to_string(++tempCounter);
    }

public:
    std::vector<std::string> generateTAC(const ASTNode* node) {
        std::vector<std::string> code;
        std::string result = generateTACHelper(node, code);
        if (!result.empty()) {
            code.push_back(result);
        }
        return code;
    }

private:
    std::string generateTACHelper(const ASTNode* node, std::vector<std::string>& code) {
        if (node->type == NodeType::INTEGER) {
            const auto* intNode = static_cast<const IntegerNode*>(node);
            return std::to_string(intNode->value);
        } else if (node->type == NodeType::VARIABLE) {
            const auto* varNode = static_cast<const VariableNode*>(node);
            return varNode->name;
        } else if (node->type == NodeType::BINARY_OP) {
            const auto* binOpNode = static_cast<const BinaryOpNode*>(node);
            std::string leftTemp = generateTACHelper(binOpNode->left.get(), code);
            std::string rightTemp = generateTACHelper(binOpNode->right.get(), code);

            std::string op;
            switch (binOpNode->op) {
                case TokenType::PLUS: op = "+"; break;
                case TokenType::MINUS: op = "-"; break;
                case TokenType::MULTIPLY: op = "*"; break;
                case TokenType::DIVIDE: op = "/"; break;
                default: throw std::runtime_error("Unknown operator");
            }

            std::string result = generateTemp();
            code.push_back(result + " = " + leftTemp + " " + op + " " + rightTemp);
            return result;
        } else if (node->type == NodeType::ASSIGNMENT) {
            const auto* assignNode = static_cast<const AssignmentNode*>(node);
            std::string exprTemp = generateTACHelper(assignNode->expression.get(), code);
            code.push_back(assignNode->variable + " = " + exprTemp);
            return "";
        } else if (node->type == NodeType::RETURN) {
            const auto* returnNode = static_cast<const ReturnNode*>(node);
            std::string exprTemp = generateTACHelper(returnNode->expression.get(), code);
            code.push_back("return " + exprTemp);
            return "";
        }
        throw std::runtime_error("Unknown node type");
    }
};

class CodeOptimizer {
public:
    std::vector<std::string> optimize(const std::vector<std::string>& tac) {
        std::vector<std::string> optimizedCode;
        std::unordered_map<std::string, std::string> constantMap;

        for (const auto& instruction : tac) {
            std::string optimizedInstruction = foldConstants(instruction, constantMap);
            if (!optimizedInstruction.empty()) {
                optimizedCode.push_back(optimizedInstruction);
            }
        }

        return optimizedCode;
    }

private:
    std::string foldConstants(const std::string& instruction, std::unordered_map<std::string, std::string>& constantMap) {
        std::istringstream iss(instruction);
        std::string result, eq, op1, op, op2;
        iss >> result >> eq;

        if (eq == "=") {
            iss >> op1;
            if (iss >> op >> op2) {

                op1 = constantMap.count(op1) ? constantMap[op1] : op1;
                op2 = constantMap.count(op2) ? constantMap[op2] : op2;

                if (isInteger(op1) && isInteger(op2)) {
                    int val1 = std::stoi(op1);
                    int val2 = std::stoi(op2);
                    int res;

                    if (op == "+") res = val1 + val2;
                    else if (op == "-") res = val1 - val2;
                    else if (op == "*") res = val1 * val2;
                    else if (op == "/") res = val1 / val2;
                    else return instruction;

                    constantMap[result] = std::to_string(res);
                    return result + " = " + std::to_string(res);
                }
            } else {

                if (isInteger(op1)) {
                    constantMap[result] = op1;
                } else if (constantMap.count(op1)) {
                    return result + " = " + constantMap[op1];
                }
            }
        } else if (result == "return") {
            iss >> op1;
            if (constantMap.count(op1)) {
                return "return " + constantMap[op1];
            }
        }

        return instruction;
    }

    bool isInteger(const std::string& s) {
        return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
    }
};

class AssemblyCodeGenerator {
private:
    std::vector<std::string> assembly;
    std::unordered_map<std::string, int> variableOffsets;
    std::unordered_set<std::string> allocatedVariables;
    int nextOffset = 4;
    std::string getVariableOffset(const std::string& var) {
        if (variableOffsets.find(var) == variableOffsets.end() && !isTemporary(var)) {
            if (allocatedVariables.find(var) == allocatedVariables.end()) {
                variableOffsets[var] = nextOffset;
                allocatedVariables.insert(var);
                nextOffset += 4;
            }
        }
        return (variableOffsets.find(var) != variableOffsets.end())
               ? "dword [ebp - " + std::to_string(variableOffsets[var]) + "]"
               : var;
    }

    bool isTemporary(const std::string& var) {
        return var[0] == 't';
    }

    bool isInteger(const std::string& s) {
        return !s.empty() && s.find_first_not_of("0123456789-") == std::string::npos;
    }

public:
    std::vector<std::string> generateAssembly(const std::vector<std::string>& tac) {
    assembly.clear();
    variableOffsets.clear();
    allocatedVariables.clear();
    nextOffset = 4;

    assembly.push_back("section .text");
    assembly.push_back("global _start");
    assembly.push_back("_start:");
    assembly.push_back("    push ebp");
    assembly.push_back("    mov ebp, esp");

    for (const auto& instruction : tac) {
        std::istringstream iss(instruction);
        std::string result, eq, op1, op, op2;
        iss >> result >> eq;
        if (eq == "=" && !isTemporary(result)) {
            getVariableOffset(result);
        }
        if (iss >> op1) {
            if (!isTemporary(op1) && !isInteger(op1)) {
                getVariableOffset(op1);
            }
            if (iss >> op >> op2) {
                if (!isTemporary(op2) && !isInteger(op2)) {
                    getVariableOffset(op2);
                }
            }
        }
    }

    int stackSize = nextOffset - 4;
    assembly.push_back("    sub esp, " + std::to_string(stackSize));

    std::unordered_set<std::string> usedVariables;
    std::string lastResult;

    for (const auto& instruction : tac) {
        std::istringstream iss(instruction);
        std::string result, eq, op1, op, op2;
        iss >> result >> eq;

        if (result == "return") {
            iss >> op1;
            if (isInteger(op1)) {
                assembly.push_back("    mov eax, " + op1);
            } else if (allocatedVariables.find(op1) != allocatedVariables.end()) {
                assembly.push_back("    mov eax, " + getVariableOffset(op1));
            }
            break;
        } else if (eq == "=") {
            iss >> op1;
            if (iss >> op >> op2) {
                if (op == "+") {
                    if (isInteger(op1) && isInteger(op2)) {
                        int value = std::stoi(op1) + std::stoi(op2);
                        assembly.push_back("    mov eax, " + std::to_string(value));
                    } else {
                        if (!isInteger(op1)) {
                            assembly.push_back("    mov eax, " + getVariableOffset(op1));
                        } else {
                            assembly.push_back("    mov eax, " + op1);
                        }
                        if (isInteger(op2)) {
                            assembly.push_back("    add eax, " + op2);
                        } else {
                            assembly.push_back("    add eax, " + getVariableOffset(op2));
                        }
                    }
                } else if (op == "-") {
                    if (isInteger(op1) && isInteger(op2)) {
                        int value = std::stoi(op1) - std::stoi(op2);
                        assembly.push_back("    mov eax, " + std::to_string(value));
                    } else {
                        if (!isInteger(op1)) {
                            assembly.push_back("    mov eax, " + getVariableOffset(op1));
                        } else {
                            assembly.push_back("    mov eax, " + op1);
                        }
                        if (isInteger(op2)) {
                            assembly.push_back("    sub eax, " + op2);
                        } else {
                            assembly.push_back("    sub eax, " + getVariableOffset(op2));
                        }
                    }
                } else if (op == "*") {
                    if (isInteger(op1) && isInteger(op2)) {
                        int value = std::stoi(op1) * std::stoi(op2);
                        assembly.push_back("    mov eax, " + std::to_string(value));
                    } else {
                        if (!isInteger(op1)) {
                            assembly.push_back("    mov eax, " + getVariableOffset(op1));
                        } else {
                            assembly.push_back("    mov eax, " + op1);
                        }
                        if (isInteger(op2)) {
                            assembly.push_back("    imul eax, " + op2);
                        } else {
                            assembly.push_back("    imul eax, " + getVariableOffset(op2));
                        }
                    }
                } else if (op == "/") {
                    if (isInteger(op1) && isInteger(op2)) {
                        int value = std::stoi(op1) / std::stoi(op2);
                        assembly.push_back("    mov eax, " + std::to_string(value));
                    } else {
                        if (!isInteger(op1)) {
                            assembly.push_back("    mov eax, " + getVariableOffset(op1));
                        } else {
                            assembly.push_back("    mov eax, " + op1);
                        }
                        if (isInteger(op2)) {
                            assembly.push_back("    mov ebx, " + op2);
                        } else {
                            assembly.push_back("    mov ebx, " + getVariableOffset(op2));
                        }
                        assembly.push_back("    xor edx, edx");
                        assembly.push_back("    div ebx");
                    }
                }
            } else {
                if (isInteger(op1)) {
                    assembly.push_back("    mov eax, " + op1);
                } else {
                    assembly.push_back("    mov eax, " + getVariableOffset(op1));
                }
            }
            if (!isTemporary(result)) {
                usedVariables.insert(result);
                assembly.push_back("    mov " + getVariableOffset(result) + ", eax");
            }
            lastResult = result;
        }
    }


        assembly.push_back("    mov esp, ebp");
    assembly.push_back("    pop ebp");
    assembly.push_back("    ret");

    return assembly;
}

};

class Compiler {
private:
    std::string sourceCode;

    std::string preprocess() {
        std::istringstream stream(sourceCode);
        std::ostringstream result;
        std::string line;

        while (std::getline(stream, line)) {
            size_t commentPos = line.find("//");
            if (commentPos != std::string::npos) {
                line = line.substr(0, commentPos);
            }
            result << line << '\n';
        }

        return result.str();
    }

    std::vector<Token> lexicalAnalysis(const std::string& preprocessedCode) {
        Lexer lexer(preprocessedCode);
        std::vector<Token> tokens;
        Token token = lexer.getNextToken();

        while (token.type != TokenType::EOF_TOKEN) {
            tokens.push_back(token);
            token = lexer.getNextToken();
        }

        return tokens;
    }

    std::vector<std::unique_ptr<ASTNode>> syntaxAnalysis(const std::string& preprocessedCode) {
        Parser parser(preprocessedCode);
        return parser.parse();
    }

    void semanticAnalysis(const std::vector<std::unique_ptr<ASTNode>>& ast) {
        SemanticAnalyzer analyzer;
        for (const auto& node : ast) {
            analyzer.analyze(node.get());
        }
    }

    std::vector<std::string> generateIntermediateCode(const std::vector<std::unique_ptr<ASTNode>>& ast) {
        IntermediateCodeGenerator icGenerator;
        std::vector<std::string> tac;
        for (const auto& node : ast) {
            auto nodeTac = icGenerator.generateTAC(node.get());
            tac.insert(tac.end(), nodeTac.begin(), nodeTac.end());
        }
        return tac;
    }

    std::vector<std::string> optimizeCode(const std::vector<std::string>& tac) {
        CodeOptimizer optimizer;
        return optimizer.optimize(tac);
    }

    std::vector<std::string> generateAssembly(const std::vector<std::string>& optimizedTac) {
        AssemblyCodeGenerator assemblyGenerator;
        return assemblyGenerator.generateAssembly(optimizedTac);
    }

public:
    explicit Compiler(const std::string& source) : sourceCode(source) {}

    void compile() {
        try {
            std::string preprocessedCode = preprocess();
            std::cout << "Preprocessed Code:\n" << preprocessedCode << "\n";

            std::vector<Token> tokens = lexicalAnalysis(preprocessedCode);
            std::cout << "Tokens:\n";
            for (const auto& token : tokens) {
                std::cout << "(" << static_cast<int>(token.type) << ", " << token.value << ") ";
            }
            std::cout << "\n\n";

            std::vector<std::unique_ptr<ASTNode>> ast = syntaxAnalysis(preprocessedCode);
            std::cout << "AST:\n";
            for (const auto& node : ast) {
                std::cout << node->toString() << "\n";
            }
            std::cout << "\n";

            semanticAnalysis(ast);
            std::cout << "Semantic analysis completed\n";

            std::vector<std::string> tac = generateIntermediateCode(ast);
            std::cout << "\nThree-Address Code:\n";
            for (const auto& instruction : tac) {
                std::cout << instruction << "\n";
            }
            std::cout << "\n";

            std::vector<std::string> optimizedTac = optimizeCode(tac);
            std::cout << "Optimized Three-Address Code:\n";
            for (const auto& instruction : optimizedTac) {
                std::cout << instruction << "\n";
            }
            std::cout << "\n";

            std::vector<std::string> assembly = generateAssembly(optimizedTac);
            std::cout << "Assembly Code:\n";
            for (const auto& instruction : assembly) {
                std::cout << instruction << "\n";
            }

        } catch (const std::exception& e) {
            std::cerr << "Compilation error: " << e.what() << std::endl;
        }
    }
};

int main() {
    std::string sourceCode = R"(
        x=2;
        return x;
    )";

    Compiler compiler(sourceCode);
    compiler.compile();

    return 0;
}
