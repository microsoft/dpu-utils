using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace MSRC.DPU.CSharpSourceGraphExtraction.GraphBuilders
{
    internal class ReturnToGraphBuilder : CSharpSyntaxWalker
    {
        private readonly SourceGraph _graph;
        private readonly Stack<SyntaxNodeOrToken> _returningPoint;

        private ReturnToGraphBuilder(SourceGraph graph)
        {
            _graph = graph;
            _returningPoint = new Stack<SyntaxNodeOrToken>();
        }

        public static void AddReturnToGraph(SourceGraph graph)
        {
            new ReturnToGraphBuilder(graph).Visit(graph.SemanticModel.SyntaxTree.GetRoot());
        }

        public override void VisitMethodDeclaration(MethodDeclarationSyntax node)
        {
            if (node.Body == null && node.ExpressionBody == null)
            {
                return; // Don't bother with abstract methods
            }

            _returningPoint.Push(node.Identifier);
            base.VisitMethodDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitConstructorDeclaration(ConstructorDeclarationSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitConstructorDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitDestructorDeclaration(DestructorDeclarationSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitDestructorDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitConversionOperatorDeclaration(ConversionOperatorDeclarationSyntax node)
        {
            _returningPoint.Push(node);
            base.VisitConversionOperatorDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitOperatorDeclaration(OperatorDeclarationSyntax node)
        {
            _returningPoint.Push(node.OperatorToken);
            base.VisitOperatorDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitPropertyDeclaration(PropertyDeclarationSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitPropertyDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitIndexerDeclaration(IndexerDeclarationSyntax node)
        {
            _returningPoint.Push(node);
            base.VisitIndexerDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitEventDeclaration(EventDeclarationSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitEventDeclaration(node);
            _returningPoint.Pop();
        }

        public override void VisitSimpleLambdaExpression(SimpleLambdaExpressionSyntax node)
        {
            _returningPoint.Push(node);
            base.VisitSimpleLambdaExpression(node);
            _returningPoint.Pop();
        }

        public override void VisitParenthesizedLambdaExpression(ParenthesizedLambdaExpressionSyntax node)
        {
            _returningPoint.Push(node);
            base.VisitParenthesizedLambdaExpression(node);
            _returningPoint.Pop();
        }

        public override void VisitLocalFunctionStatement(LocalFunctionStatementSyntax node)
        {
            _returningPoint.Push(node.Identifier);
            base.VisitLocalFunctionStatement(node);
            _returningPoint.Pop();
        }

        public override void VisitAnonymousMethodExpression(AnonymousMethodExpressionSyntax node)
        {
            _returningPoint.Push(node);
            base.VisitAnonymousMethodExpression(node);
            _returningPoint.Pop();
        }

        public override void VisitReturnStatement(ReturnStatementSyntax node)
        {
            Debug.Assert(_returningPoint.Count > 0);
            if (node.Expression != null)
            {
                _graph.AddEdge(node.Expression, SourceGraphEdge.ReturnsTo, _returningPoint.Peek());
            }
            base.VisitReturnStatement(node);
        }

        public override void VisitYieldStatement(YieldStatementSyntax node)
        {
            bool isReturnStatement = node.IsKind(SyntaxKind.YieldReturnStatement);
            if (isReturnStatement && node.Expression != null)
            {
                _graph.AddEdge(node.Expression, SourceGraphEdge.ReturnsTo, _returningPoint.Peek());
            }
            base.VisitYieldStatement(node);
        }
    }
}
