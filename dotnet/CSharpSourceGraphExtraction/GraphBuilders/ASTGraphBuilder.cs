using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System.Linq;

namespace MSRC.DPU.CSharpSourceGraphExtraction.GraphBuilders
{
    internal class ASTGraphBuilder : CSharpSyntaxWalker
    {
        private readonly SourceGraph _graph;
        private SyntaxToken? _lastAddedToken;

        private ASTGraphBuilder(SourceGraph graph)
        {
            _graph = graph;
        }

        public static void AddASTGraph(SourceGraph sourceGraph)
        {
            new ASTGraphBuilder(sourceGraph).Visit(sourceGraph.SemanticModel.SyntaxTree.GetRoot());
        }

        private void AddToken(SyntaxToken token)
        {
            if (_lastAddedToken.HasValue)
            {
                _graph.AddEdge(_lastAddedToken.Value, SourceGraphEdge.NextToken, token);
            }
            _lastAddedToken = token;
        }

        public override void Visit(SyntaxNode node)
        {
            foreach (var child in node.ChildNodesAndTokens())
            {
                if (!node.DescendantNodes().Any(n=>n is BaseMethodDeclarationSyntax || n is PropertyDeclarationSyntax))
                {
                    _graph.AddEdge(node, SourceGraphEdge.Child, child);
                }                
                if (child.IsNode)
                {
                    Visit(child.AsNode());
                }
                else
                {
                    AddToken(child.AsToken());
                }
            }
        }
    }
}
