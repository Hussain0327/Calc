#!/usr/bin/env python3
"""
Advanced Scientific Calculator with Expression Parser
Supports complex mathematical operations, functions, and constants
"""

import math
import re
from typing import Dict, Union, Callable, Optional
from dataclasses import dataclass
from enum import Enum

class TokenType(Enum):
    NUMBER = "NUMBER"
    OPERATOR = "OPERATOR"
    FUNCTION = "FUNCTION"
    CONSTANT = "CONSTANT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"

@dataclass
class Token:
    type: TokenType
    value: Union[str, float]
    position: int

class MathParser:
    """Recursive descent parser for mathematical expressions"""
    
    def __init__(self):
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
            'inf': float('inf'),
        }
        
        self.functions = {
            # Trigonometric
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            
            # Logarithmic
            'log': math.log10,
            'ln': math.log,
            'log2': math.log2,
            
            # Power & Root
            'sqrt': math.sqrt,
            'cbrt': lambda x: x ** (1/3),
            'exp': math.exp,
            
            # Rounding
            'floor': math.floor,
            'ceil': math.ceil,
            'round': round,
            'abs': abs,
            
            # Statistical
            'factorial': math.factorial,
            'gcd': math.gcd,
            'lcm': math.lcm,
            
            # Special
            'deg': math.degrees,
            'rad': math.radians,
        }
        
        self.binary_functions = {
            'pow': math.pow,
            'mod': lambda x, y: x % y,
            'atan2': math.atan2,
            'hypot': math.hypot,
            'gcd': math.gcd,
            'lcm': math.lcm,
        }
    
    def tokenize(self, expression: str) -> list[Token]:
        """Convert expression string into tokens"""
        tokens = []
        i = 0
        expression = expression.replace(' ', '').lower()
        
        while i < len(expression):
            # Numbers (including decimals and scientific notation)
            if expression[i].isdigit() or (expression[i] == '.' and i+1 < len(expression) and expression[i+1].isdigit()):
                j = i
                has_dot = False
                has_e = False
                
                while j < len(expression):
                    if expression[j].isdigit():
                        j += 1
                    elif expression[j] == '.' and not has_dot and not has_e:
                        has_dot = True
                        j += 1
                    elif expression[j] in 'e' and not has_e and j+1 < len(expression):
                        has_e = True
                        j += 1
                        if expression[j] in '+-':
                            j += 1
                    else:
                        break
                
                tokens.append(Token(TokenType.NUMBER, float(expression[i:j]), i))
                i = j
                
            # Functions and constants
            elif expression[i].isalpha():
                j = i
                while j < len(expression) and expression[j].isalpha():
                    j += 1
                
                word = expression[i:j]
                if word in self.constants:
                    tokens.append(Token(TokenType.CONSTANT, word, i))
                elif word in self.functions or word in self.binary_functions:
                    tokens.append(Token(TokenType.FUNCTION, word, i))
                else:
                    raise ValueError(f"Unknown identifier: {word}")
                i = j
                
            # Operators and parentheses
            elif expression[i] in '+-*/^%':
                tokens.append(Token(TokenType.OPERATOR, expression[i], i))
                i += 1
            elif expression[i] == '(':
                tokens.append(Token(TokenType.LPAREN, '(', i))
                i += 1
            elif expression[i] == ')':
                tokens.append(Token(TokenType.RPAREN, ')', i))
                i += 1
            elif expression[i] == ',':
                tokens.append(Token(TokenType.COMMA, ',', i))
                i += 1
            else:
                raise ValueError(f"Unexpected character at position {i}: {expression[i]}")
        
        return tokens
    
    def parse(self, expression: str) -> float:
        """Parse and evaluate mathematical expression"""
        self.tokens = self.tokenize(expression)
        self.pos = 0
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.tokens[self.pos].position}")
        
        return result
    
    def _current_token(self) -> Optional[Token]:
        """Get current token without consuming"""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    
    def _consume_token(self) -> Optional[Token]:
        """Get current token and advance position"""
        token = self._current_token()
        if token:
            self.pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """Parse additive expression (lowest precedence)"""
        left = self._parse_term()
        
        while self._current_token() and self._current_token().type == TokenType.OPERATOR:
            if self._current_token().value in '+-':
                op = self._consume_token().value
                right = self._parse_term()
                left = left + right if op == '+' else left - right
            else:
                break
        
        return left
    
    def _parse_term(self) -> float:
        """Parse multiplicative expression"""
        left = self._parse_power()
        
        while self._current_token() and self._current_token().type == TokenType.OPERATOR:
            if self._current_token().value in '*/%':
                op = self._consume_token().value
                right = self._parse_power()
                if op == '*':
                    left = left * right
                elif op == '/':
                    if right == 0:
                        raise ValueError("Division by zero")
                    left = left / right
                else:  # %
                    left = left % right
            else:
                break
        
        return left
    
    def _parse_power(self) -> float:
        """Parse power expression (highest precedence for operators)"""
        left = self._parse_factor()
        
        while self._current_token() and self._current_token().type == TokenType.OPERATOR:
            if self._current_token().value == '^':
                self._consume_token()
                right = self._parse_power()  # Right associative
                left = left ** right
            else:
                break
        
        return left
    
    def _parse_factor(self) -> float:
        """Parse individual factors (numbers, functions, parentheses)"""
        token = self._current_token()
        
        if not token:
            raise ValueError("Unexpected end of expression")
        
        # Unary minus
        if token.type == TokenType.OPERATOR and token.value == '-':
            self._consume_token()
            return -self._parse_factor()
        
        # Unary plus
        if token.type == TokenType.OPERATOR and token.value == '+':
            self._consume_token()
            return self._parse_factor()
        
        # Numbers
        if token.type == TokenType.NUMBER:
            self._consume_token()
            return token.value
        
        # Constants
        if token.type == TokenType.CONSTANT:
            self._consume_token()
            return self.constants[token.value]
        
        # Functions
        if token.type == TokenType.FUNCTION:
            func_name = token.value
            self._consume_token()
            
            if self._current_token() and self._current_token().type == TokenType.LPAREN:
                self._consume_token()  # Consume '('
                
                # Check if it's a binary function
                if func_name in self.binary_functions:
                    arg1 = self._parse_expression()
                    if not self._current_token() or self._current_token().type != TokenType.COMMA:
                        raise ValueError(f"Function {func_name} requires two arguments")
                    self._consume_token()  # Consume ','
                    arg2 = self._parse_expression()
                    
                    if not self._current_token() or self._current_token().type != TokenType.RPAREN:
                        raise ValueError("Expected ')'")
                    self._consume_token()  # Consume ')'
                    
                    return self.binary_functions[func_name](arg1, arg2)
                else:
                    arg = self._parse_expression()
                    
                    if not self._current_token() or self._current_token().type != TokenType.RPAREN:
                        raise ValueError("Expected ')'")
                    self._consume_token()  # Consume ')'
                    
                    return self.functions[func_name](arg)
            else:
                raise ValueError(f"Expected '(' after function {func_name}")
        
        # Parentheses
        if token.type == TokenType.LPAREN:
            self._consume_token()  # Consume '('
            result = self._parse_expression()
            
            if not self._current_token() or self._current_token().type != TokenType.RPAREN:
                raise ValueError("Mismatched parentheses")
            self._consume_token()  # Consume ')'
            
            return result
        
        raise ValueError(f"Unexpected token: {token.value}")

class Calculator:
    """Main calculator interface with history and advanced features"""
    
    def __init__(self):
        self.parser = MathParser()
        self.history = []
        self.variables = {}
        self.last_result = 0
    
    def evaluate(self, expression: str) -> float:
        """Evaluate expression with variable substitution"""
        # Replace 'ans' with last result
        expression = expression.replace('ans', str(self.last_result))
        
        # Replace variables
        for var, value in self.variables.items():
            expression = expression.replace(var, str(value))
        
        result = self.parser.parse(expression)
        self.last_result = result
        self.history.append((expression, result))
        
        return result
    
    def set_variable(self, name: str, value: float):
        """Store a variable for later use"""
        if name in self.parser.constants or name in self.parser.functions:
            raise ValueError(f"Cannot use reserved name: {name}")
        self.variables[name] = value
    
    def show_history(self, n: int = 10):
        """Display last n calculations"""
        for expr, result in self.history[-n:]:
            print(f"  {expr} = {result}")
    
    def clear_history(self):
        """Clear calculation history"""
        self.history = []
        self.last_result = 0

def main():
    calc = Calculator()
    
    print("=" * 60)
    print("ADVANCED SCIENTIFIC CALCULATOR")
    print("=" * 60)
    print()
    print("Features:")
    print("  • Basic: +, -, *, /, ^, %")
    print("  • Functions: sin, cos, tan, log, ln, sqrt, exp, etc.")
    print("  • Constants: pi, e, tau, phi")
    print("  • Variables: x=value to store, 'ans' for last result")
    print("  • Examples: 2^10, sin(pi/2), sqrt(2), log(100)")
    print()
    print("Commands:")
    print("  help     - Show this help")
    print("  vars     - Show stored variables")
    print("  history  - Show calculation history")
    print("  clear    - Clear history and variables")
    print("  quit     - Exit calculator")
    print("=" * 60)
    print()
    
    while True:
        try:
            user_input = input("calc> ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nAvailable functions:")
                for func in sorted(calc.parser.functions.keys()):
                    print(f"  {func}", end="  ")
                print("\n\nAvailable constants:")
                for const in sorted(calc.parser.constants.keys()):
                    print(f"  {const} = {calc.parser.constants[const]}")
                print()
            elif user_input.lower() == 'vars':
                if calc.variables:
                    print("\nStored variables:")
                    for var, value in calc.variables.items():
                        print(f"  {var} = {value}")
                else:
                    print("No variables stored")
                print()
            elif user_input.lower() == 'history':
                if calc.history:
                    print("\nRecent calculations:")
                    calc.show_history()
                else:
                    print("No history yet")
                print()
            elif user_input.lower() == 'clear':
                calc.clear_history()
                calc.variables = {}
                print("Cleared history and variables\n")
            
            # Handle variable assignment
            elif '=' in user_input and not any(op in user_input.split('=')[0] for op in '+-*/^%()<>!'):
                parts = user_input.split('=')
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    var_value = calc.evaluate(parts[1].strip())
                    calc.set_variable(var_name, var_value)
                    print(f"{var_name} = {var_value}\n")
                else:
                    print("Invalid assignment\n")
            
            # Evaluate expression
            else:
                result = calc.evaluate(user_input)
                
                # Format output based on magnitude
                if abs(result) < 1e-10 and result != 0:
                    print(f"= {result:.2e}\n")
                elif abs(result) > 1e10:
                    print(f"= {result:.2e}\n")
                elif result == int(result):
                    print(f"= {int(result)}\n")
                else:
                    print(f"= {result}\n")
                    
        except ValueError as e:
            print(f"Error: {e}\n")
        except Exception as e:
            print(f"Unexpected error: {e}\n")

if __name__ == "__main__":
    main()