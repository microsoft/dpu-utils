using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using MSRC.DPU.CSharpSourceGraphExtraction.Utils;

namespace MSRC.DPU.CSharpSourceGraphExtraction
{
    internal class MethodUseInformationCollector : CSharpSyntaxWalker
    {
        private readonly SourceGraph _graph;

        private MethodUseInformationCollector(SourceGraph graph)
        {
            _graph = graph;
        }

        public static void AddMethodUseInformation(SourceGraph graph)
        {
            new MethodUseInformationCollector(graph).Visit(graph.SemanticModel.SyntaxTree.GetRoot());
        }

        private void RecordDeclaration(SyntaxNode declarationNode, ParameterListSyntax parameterList)
        {
            var methodSymbol = _graph.SemanticModel.GetDeclaredSymbol(declarationNode) as IMethodSymbol;
            if (methodSymbol == null || methodSymbol.IsAbstract)
            {
                return; // Skip abstract methods.
            }
            var declarationInfo = new MethodDeclarationInformation(
                methodSymbol,
                declarationNode,
                ParamSymbolsToTokens(parameterList, methodSymbol));
            _graph.MethodDeclarationSites.Add(declarationInfo);
        }

        private void RecordInvocation(IMethodSymbol invokedMethodSymbol, SyntaxNode methodInvocationNode, ArgumentListSyntax methodArguments)
        {
            var argumentNodeToMethodParameter = new Dictionary<SyntaxNodeOrToken, (string parameterName, ITypeSymbol parameterType)>();
            if (methodArguments != null)
            {
                foreach (var arg in methodArguments.Arguments)
                {
                    var paramSymbol = MethodUtils.DetermineParameter(methodArguments, arg, invokedMethodSymbol);
                    Debug.Assert(arg.Expression != null);
                    Debug.Assert(paramSymbol != null);
                    argumentNodeToMethodParameter.Add(arg.Expression, (paramSymbol.Name, paramSymbol.Type));
                }
            }

            var invocationInfo = new MethodInvocationInformation(invokedMethodSymbol, methodInvocationNode, argumentNodeToMethodParameter);
            _graph.MethodInvocationSites.Add(invocationInfo);
        }

        private static Dictionary<IParameterSymbol, SyntaxToken> ParamSymbolsToTokens(ParameterListSyntax parameterList, IMethodSymbol methodSymbol)
        {
            Debug.Assert(methodSymbol != null);
            var paramSymbolsToIdentifiers = new Dictionary<IParameterSymbol, SyntaxToken>();
            if (parameterList == null || parameterList.Parameters == null || methodSymbol.Parameters.Length != parameterList.Parameters.Count)
            {
                return paramSymbolsToIdentifiers;
            }
            for (int i = 0; i < methodSymbol.Parameters.Length; i++)
            {
                paramSymbolsToIdentifiers.Add(methodSymbol.Parameters[i], parameterList.Parameters[i].Identifier);
            }
            return paramSymbolsToIdentifiers;
        }

        public override void VisitInvocationExpression(InvocationExpressionSyntax node)
        {
            var invocationSymbol = _graph.SemanticModel.GetSymbolInfo(node).Symbol as IMethodSymbol;

            if (invocationSymbol == null) return;
            string methodName = invocationSymbol.Name;
            if (invocationSymbol.Name.IndexOf('<') != -1)
            {
                methodName = methodName.Substring(0, methodName.IndexOf('<'));
            }
            string invocationExpression;
            if (node.Expression is MemberAccessExpressionSyntax memberAccess)
            {
                invocationExpression = memberAccess.Name.ToString();
            } else
            {
                invocationExpression = node.Expression.ToString();
            }
            if (invocationExpression.IndexOf('<') != -1)
            {
                invocationExpression = invocationExpression.Substring(0, invocationExpression.IndexOf('<'));
            }

            if (!invocationExpression.EndsWith(methodName))
            {
                // Heuristic: this may happen when an implicit conversion exits e.g. "string x = SomeObjectRetType()"
                // or an invocation of an anonymous function, such as a lambda, when the method name is "Invoke"
                if (methodName != "Invoke")
                {
                    Console.WriteLine($"Rejecting Invocation because expression name and symbol do not match: {methodName} -> {node.Expression}");
                }
                return; 
            }
            else if (node.ArgumentList.ToString().Contains("__arglist"))
            {
                Console.WriteLine($"Rejecting Invocation because it contains an __arglist: {node}");
                return;
            }

            RecordInvocation(invocationSymbol, node, node.ArgumentList);
            base.VisitInvocationExpression(node);
        }

        public override void VisitObjectCreationExpression(ObjectCreationExpressionSyntax node)
        {
            var constructorSymbol = _graph.SemanticModel.GetSymbolInfo(node).Symbol as IMethodSymbol;
            if (constructorSymbol == null) return;

            RecordInvocation(constructorSymbol, node, node.ArgumentList);
            base.VisitObjectCreationExpression(node);
        }

        public override void VisitMethodDeclaration(MethodDeclarationSyntax node)
        {
            if (node.Body == null && node.ExpressionBody == null)
            {
                return; // Method must be abstract
            }

            RecordDeclaration(node, node.ParameterList);
            base.VisitMethodDeclaration(node);
        }

        public override void VisitLocalFunctionStatement(LocalFunctionStatementSyntax node)
        {
            RecordDeclaration(node, node.ParameterList);
            base.VisitLocalFunctionStatement(node);
        }
    }
}
