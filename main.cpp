#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

enum class TokenType {
    INTEGER,
    PLUS,
    MINUS,
    MULTIPLY,
    DIVIDE,
    LPAREN,
    RPAREN,
    SEMICOLON,
    EOF_TOKEN
};

struct Token {
    TokenType type;
    std::string value;
    Token(TokenType t, const std::string& v) : type(t), value(v) {}
};

enum class NodeType {
    BINARY_OP,
    INTEGER
};

class ASTNode {
public:
    NodeType type;
    explicit ASTNode(NodeType t) : type(t) {}
    virtual ~ASTNode() = default;
};

class IntegerNode : public ASTNode {
public:
    int value;
    IntegerNode(int val) : ASTNode(NodeType::INTEGER), value(val) {}
};

class BinaryOpNode : public ASTNode {
public:
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;
    TokenType op;
    BinaryOpNode(std::unique_ptr<ASTNode> l, std::unique_ptr<ASTNode> r, TokenType o)
        : ASTNode(NodeType::BINARY_OP), left(std::move(l)), right(std::move(r)), op(o) {}
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
            default:
                throw std::runtime_error("Invalid character encountered");
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
            throw std::runtime_error("Unexpected token");
        }
    }

    std::unique_ptr<ASTNode> factor() {
        if (currentToken.type == TokenType::INTEGER) {
            int value = std::stoi(currentToken.value);
            eat(TokenType::INTEGER);
            return std::make_unique<IntegerNode>(value);
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

    std::unique_ptr<ASTNode> statement() {
        auto node = expr();
        eat(TokenType::SEMICOLON);
        return node;
    }

public:
    explicit Parser(const std::string& input) : lexer(input), currentToken(TokenType::EOF_TOKEN, "") {
        currentToken = lexer.getNextToken();
    }

    std::unique_ptr<ASTNode> parse() {
        auto node = statement();
        if (currentToken.type != TokenType::EOF_TOKEN) {
            throw std::runtime_error("Unexpected tokens after statement");
        }
        return node;
    }
};

class SemanticAnalyzer {
public:
    void analyze(const ASTNode* node) {
        if (node->type == NodeType::INTEGER) {
        } else if (node->type == NodeType::BINARY_OP) {
            const auto* binOp = static_cast<const BinaryOpNode*>(node);
            analyze(binOp->left.get());
            analyze(binOp->right.get());
        }
    }
};

void printAST(const ASTNode* node, int level = 0) {
    if (node->type == NodeType::INTEGER) {
        const auto* intNode = static_cast<const IntegerNode*>(node);
        std::cout << std::string(level * 2, ' ') << "Integer: " << intNode->value << std::endl;
    } else if (node->type == NodeType::BINARY_OP) {
        const auto* binOpNode = static_cast<const BinaryOpNode*>(node);
        std::cout << std::string(level * 2, ' ') << "BinaryOp: " << static_cast<int>(binOpNode->op) << std::endl;
        std::cout << std::string(level * 2, ' ') << "Left:" << std::endl;
        printAST(binOpNode->left.get(), level + 1);
        std::cout << std::string(level * 2, ' ') << "Right:" << std::endl;
        printAST(binOpNode->right.get(), level + 1);
    }
}

class IntermediateCodeGenerator {
public:
    std::vector<std::string> generateTAC(const ASTNode* node) {
        std::vector<std::string> code;
        generateTACHelper(node, code);
        return code;
    }

private:
    int tempCounter = 0;

    std::string generateTemp() {
        return "t" + std::to_string(++tempCounter);
    }

    std::string generateTACHelper(const ASTNode* node, std::vector<std::string>& code) {
        if (node->type == NodeType::INTEGER) {
            const auto* intNode = static_cast<const IntegerNode*>(node);
            std::string temp = generateTemp();
            code.push_back(temp + " = " + std::to_string(intNode->value));
            return temp;
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
        }
        throw std::runtime_error("Unknown node type");
    }
};

class CodeOptimizer {
public:
    std::vector<std::string> optimize(const std::vector<std::string>& tac) {
        std::vector<std::string> optimizedCode;
        for (const auto& instruction : tac) {
            std::string optimizedInstruction = foldConstants(instruction);
            optimizedCode.push_back(optimizedInstruction);
        }
        return optimizedCode;
    }

private:
    std::string foldConstants(const std::string& instruction) {
        std::istringstream iss(instruction);
        std::string result, op1, op, op2;
        iss >> result >> op >> op1 >> op >> op2;

        if (op1[0] != 't' && op2[0] != 't') {
            try{
                int val1 = std::stoi(op1);
                int val2 = std::stoi(op2);
                int res;
                if (op == "+") res = val1 + val2;
                else if (op == "-") res = val1 - val2;
                else if (op == "*") res = val1 * val2;
                else if (op == "/") res = val1 / val2;
                else return instruction;

                return result + " = " + std::to_string(res);
            }catch (const std::exception&) {
                return instruction;
            }
        }
        return instruction;
    }
};

class AssemblyCodeGenerator {
public:
    std::vector<std::string> generateAssembly(const std::vector<std::string>& tac) {
        std::vector<std::string> assembly;
        assembly.push_back("section .text");
        assembly.push_back("global _start");
        assembly.push_back("_start:");

        std::unordered_map<std::string, std::string> varToReg;
        int stackOffset = 0;

        for (const auto& instruction : tac) {
            std::istringstream iss(instruction);
            std::string result, op1, op, op2;
            iss >> result >> op >> op1 >> op >> op2;

            if (op.empty()) {
                assembly.push_back("    mov eax, " + op1);
                varToReg[result] = "eax";
            } else {
                std::string reg1 = getReg(op1, varToReg, assembly, stackOffset);
                std::string reg2 = getReg(op2, varToReg, assembly, stackOffset);

                if (op == "+") assembly.push_back("    add " + reg1 + ", " + reg2);
                else if (op == "-") assembly.push_back("    sub " + reg1 + ", " + reg2);
                else if (op == "*") assembly.push_back("    imul " + reg1 + ", " + reg2);
                else if (op == "/") {
                    assembly.push_back("    mov eax, " + reg1);
                    assembly.push_back("    cdq");
                    assembly.push_back("    idiv " + reg2);
                }

                varToReg[result] = reg1;
            }
        }

        assembly.push_back("    mov eax, 1");
        assembly.push_back("    xor ebx, ebx");
        assembly.push_back("    int 0x80");

        return assembly;
    }

private:
    std::string getReg(const std::string& var, std::unordered_map<std::string, std::string>& varToReg,
                       std::vector<std::string>& assembly, int& stackOffset) {
        if (var[0] != 't') return "$" + var;
        if (varToReg.find(var) != varToReg.end()) return varToReg[var];

        std::string newReg = "ebx";
        assembly.push_back("    mov " + newReg + ", [esp + " + std::to_string(stackOffset) + "]");
        stackOffset += 4;
        varToReg[var] = newReg;
        return newReg;
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

    std::unique_ptr<ASTNode> syntaxAnalysis(const std::string& preprocessedCode) {
        Parser parser(preprocessedCode);
        return parser.parse();
    }

    void semanticAnalysis(const ASTNode* ast) {
        SemanticAnalyzer analyzer;
        analyzer.analyze(ast);
    }

    std::vector<std::string> generateIntermediateCode(const ASTNode* ast) {
        IntermediateCodeGenerator icGenerator;
        return icGenerator.generateTAC(ast);
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
            std::cout << "Preprocessed Code:" << preprocessedCode << "\n";

            std::vector<Token> tokens = lexicalAnalysis(preprocessedCode);
            std::cout << "Tokens:\n";
            for (const auto& token : tokens) {
                std::cout << "(" << static_cast<int>(token.type) << ", " << token.value << ") ";
            }
            std::cout << "\n\n";

            std::unique_ptr<ASTNode> ast = syntaxAnalysis(preprocessedCode);
            std::cout << "AST created:\n";
            printAST(ast.get());
            std::cout << "\n";

            semanticAnalysis(ast.get());
            std::cout << "Semantic analysis completed\n";

            std::vector<std::string> tac = generateIntermediateCode(ast.get());
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
        5 + 3 * (10 - 2);
    )";

    Compiler compiler(sourceCode);
    compiler.compile();

    return 0;
}
