#!/usr/bin/env python3
"""
Graph Neural Network (GNN) scaffold for financial networks
Provides skeleton infrastructure for GNN-based portfolio analysis with lightweight data loaders
AC2: GNN scaffold defines data loaders only (no heavy dependencies)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from pathlib import Path

from config.feature_flags import is_gnn_scaffold_enabled

logger = logging.getLogger(__name__)


class GraphEdge(NamedTuple):
    """Represents an edge in the financial graph"""
    source: str
    target: str
    weight: float
    edge_type: str  # 'correlation', 'sector', 'supply_chain', etc.
    metadata: Dict[str, Any] = {}


class GraphNode(NamedTuple):
    """Represents a node in the financial graph"""
    symbol: str
    node_type: str  # 'stock', 'sector', 'index', etc.
    features: Dict[str, float] = {}
    metadata: Dict[str, Any] = {}


@dataclass
class GraphData:
    """Container for graph structure and features"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    node_features: Optional[np.ndarray] = None
    edge_features: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize derived properties"""
        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.edges)
        self.symbol_to_idx = {node.symbol: i for i, node in enumerate(self.nodes)}


class BaseDataLoader(ABC):
    """Abstract base class for GNN data loaders"""
    
    @abstractmethod
    def load_graph_data(self, symbols: List[str], **kwargs) -> Optional[GraphData]:
        """Load graph data for given symbols"""
        pass
    
    @abstractmethod
    def get_adjacency_matrix(self, graph_data: GraphData) -> np.ndarray:
        """Get adjacency matrix from graph data"""
        pass
    
    @abstractmethod
    def get_feature_matrix(self, graph_data: GraphData) -> np.ndarray:
        """Get node feature matrix from graph data"""
        pass


class CorrelationGraphLoader(BaseDataLoader):
    """
    Data loader for correlation-based financial graphs
    Creates graph where edges represent price correlations between stocks
    """
    
    def __init__(self, correlation_threshold: float = 0.3, max_lookback_days: int = 252):
        """
        Initialize correlation graph loader
        
        Args:
            correlation_threshold: Minimum correlation for edge creation
            max_lookback_days: Maximum lookback period for correlation calculation
        """
        self.correlation_threshold = correlation_threshold
        self.max_lookback_days = max_lookback_days
        
    def load_graph_data(self, symbols: List[str], 
                       price_data: Optional[Dict[str, pd.DataFrame]] = None,
                       correlation_matrix: Optional[pd.DataFrame] = None) -> Optional[GraphData]:
        """
        Load correlation-based graph data
        
        Args:
            symbols: List of stock symbols
            price_data: Optional price data for correlation calculation
            correlation_matrix: Pre-computed correlation matrix
            
        Returns:
            GraphData object or None if loading fails
        """
        if not is_gnn_scaffold_enabled():
            logger.debug("GNN scaffold disabled by feature flag")
            return None
            
        try:
            # Create nodes for each symbol
            nodes = []
            for symbol in symbols:
                node = GraphNode(
                    symbol=symbol,
                    node_type='stock',
                    features={'symbol_hash': hash(symbol) % 1000},  # Simple feature
                    metadata={'sector': 'unknown'}  # Placeholder
                )
                nodes.append(node)
            
            # Create edges based on correlation
            edges = []
            
            if correlation_matrix is not None:
                # Use provided correlation matrix
                for i, symbol1 in enumerate(symbols):
                    if symbol1 not in correlation_matrix.index:
                        continue
                        
                    for j, symbol2 in enumerate(symbols):
                        if i >= j or symbol2 not in correlation_matrix.columns:
                            continue
                            
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        
                        if abs(correlation) >= self.correlation_threshold:
                            edge = GraphEdge(
                                source=symbol1,
                                target=symbol2,
                                weight=float(correlation),
                                edge_type='correlation',
                                metadata={'abs_correlation': abs(correlation)}
                            )
                            edges.append(edge)
                            
            elif price_data is not None:
                # Compute correlation from price data
                from analytics.correlation import CorrelationAnalyzer
                
                analyzer = CorrelationAnalyzer(lookback_days=self.max_lookback_days)
                correlation_matrix = analyzer.compute_correlation_matrix(price_data)
                
                if correlation_matrix is not None:
                    # Recursive call with computed matrix
                    return self.load_graph_data(symbols, correlation_matrix=correlation_matrix)
            
            # Create placeholder node features (stocks as simple vectors)
            node_features = np.array([
                [node.features.get('symbol_hash', 0.0)]
                for node in nodes
            ], dtype=np.float32)
            
            # Create edge features (correlation strengths)
            if edges:
                edge_features = np.array([
                    [edge.weight, abs(edge.weight)]
                    for edge in edges
                ], dtype=np.float32)
            else:
                edge_features = np.empty((0, 2), dtype=np.float32)
            
            graph_data = GraphData(
                nodes=nodes,
                edges=edges,
                node_features=node_features,
                edge_features=edge_features
            )
            
            logger.info(f"Loaded correlation graph: {len(nodes)} nodes, {len(edges)} edges")
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error loading correlation graph data: {e}")
            return None
    
    def get_adjacency_matrix(self, graph_data: GraphData) -> np.ndarray:
        """
        Get weighted adjacency matrix from graph data
        
        Args:
            graph_data: Graph data container
            
        Returns:
            Weighted adjacency matrix (symmetric for undirected graph)
        """
        n = graph_data.num_nodes
        adjacency = np.zeros((n, n), dtype=np.float32)
        
        for edge in graph_data.edges:
            if edge.source in graph_data.symbol_to_idx and edge.target in graph_data.symbol_to_idx:
                i = graph_data.symbol_to_idx[edge.source]
                j = graph_data.symbol_to_idx[edge.target]
                
                # Symmetric matrix (undirected graph)
                adjacency[i, j] = edge.weight
                adjacency[j, i] = edge.weight
        
        return adjacency
    
    def get_feature_matrix(self, graph_data: GraphData) -> np.ndarray:
        """
        Get node feature matrix
        
        Args:
            graph_data: Graph data container
            
        Returns:
            Node feature matrix (num_nodes x feature_dim)
        """
        if graph_data.node_features is not None:
            return graph_data.node_features
        
        # Fallback: create identity features
        return np.eye(graph_data.num_nodes, dtype=np.float32)


class SectorGraphLoader(BaseDataLoader):
    """
    Data loader for sector-based financial graphs
    Creates graph where edges connect stocks in the same sector
    """
    
    def __init__(self, sector_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize sector graph loader
        
        Args:
            sector_mapping: Dict mapping symbols to sector names
        """
        self.sector_mapping = sector_mapping or {}
        
    def load_graph_data(self, symbols: List[str], **kwargs) -> Optional[GraphData]:
        """
        Load sector-based graph data
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            GraphData object or None
        """
        if not is_gnn_scaffold_enabled():
            logger.debug("GNN scaffold disabled by feature flag")
            return None
            
        try:
            # Create nodes with sector information
            nodes = []
            symbol_to_sector = {}
            
            for symbol in symbols:
                sector = self.sector_mapping.get(symbol, 'unknown')
                symbol_to_sector[symbol] = sector
                
                node = GraphNode(
                    symbol=symbol,
                    node_type='stock',
                    features={'sector_hash': hash(sector) % 100},
                    metadata={'sector': sector}
                )
                nodes.append(node)
            
            # Create edges between stocks in same sector
            edges = []
            
            for i, symbol1 in enumerate(symbols):
                sector1 = symbol_to_sector[symbol1]
                
                for j, symbol2 in enumerate(symbols):
                    if i >= j:
                        continue
                        
                    sector2 = symbol_to_sector[symbol2]
                    
                    if sector1 == sector2 and sector1 != 'unknown':
                        edge = GraphEdge(
                            source=symbol1,
                            target=symbol2,
                            weight=1.0,  # Binary connection
                            edge_type='sector',
                            metadata={'sector': sector1}
                        )
                        edges.append(edge)
            
            # Create node features (sector embeddings)
            unique_sectors = set(symbol_to_sector.values())
            sector_to_idx = {sector: i for i, sector in enumerate(sorted(unique_sectors))}
            
            node_features = np.zeros((len(nodes), len(unique_sectors)), dtype=np.float32)
            for i, node in enumerate(nodes):
                sector = node.metadata['sector']
                if sector in sector_to_idx:
                    node_features[i, sector_to_idx[sector]] = 1.0
            
            # Edge features (all same weight for sector connections)
            if edges:
                edge_features = np.ones((len(edges), 1), dtype=np.float32)
            else:
                edge_features = np.empty((0, 1), dtype=np.float32)
            
            graph_data = GraphData(
                nodes=nodes,
                edges=edges,
                node_features=node_features,
                edge_features=edge_features
            )
            
            logger.info(f"Loaded sector graph: {len(nodes)} nodes, {len(edges)} edges")
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error loading sector graph data: {e}")
            return None
    
    def get_adjacency_matrix(self, graph_data: GraphData) -> np.ndarray:
        """Get binary adjacency matrix for sector connections"""
        n = graph_data.num_nodes
        adjacency = np.zeros((n, n), dtype=np.float32)
        
        for edge in graph_data.edges:
            if edge.source in graph_data.symbol_to_idx and edge.target in graph_data.symbol_to_idx:
                i = graph_data.symbol_to_idx[edge.source]
                j = graph_data.symbol_to_idx[edge.target]
                
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
        
        return adjacency
    
    def get_feature_matrix(self, graph_data: GraphData) -> np.ndarray:
        """Get sector-based feature matrix"""
        if graph_data.node_features is not None:
            return graph_data.node_features
            
        return np.eye(graph_data.num_nodes, dtype=np.float32)


class GNNDataManager:
    """
    High-level manager for GNN data loading and preprocessing
    Coordinates different data loaders and provides unified interface
    """
    
    def __init__(self):
        self.loaders = {
            'correlation': CorrelationGraphLoader(),
            'sector': SectorGraphLoader()
        }
        self.graph_cache = {}
        
    def load_financial_graph(self, 
                           symbols: List[str],
                           graph_type: str = 'correlation',
                           **kwargs) -> Optional[GraphData]:
        """
        Load financial graph data using specified loader
        
        Args:
            symbols: List of stock symbols
            graph_type: Type of graph to load ('correlation', 'sector')
            **kwargs: Additional arguments for specific loader
            
        Returns:
            GraphData object or None
        """
        if not is_gnn_scaffold_enabled():
            logger.debug("GNN scaffold disabled by feature flag")
            return None
            
        if graph_type not in self.loaders:
            logger.error(f"Unknown graph type: {graph_type}")
            return None
        
        try:
            # Check cache
            cache_key = f"{graph_type}_{hash(tuple(sorted(symbols)))}"
            if cache_key in self.graph_cache:
                logger.debug(f"Using cached graph data for {graph_type}")
                return self.graph_cache[cache_key]
            
            # Load using appropriate loader
            loader = self.loaders[graph_type]
            graph_data = loader.load_graph_data(symbols, **kwargs)
            
            # Cache result
            if graph_data is not None:
                self.graph_cache[cache_key] = graph_data
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error loading {graph_type} graph: {e}")
            return None
    
    def get_graph_statistics(self, graph_data: GraphData) -> Dict[str, Any]:
        """
        Get statistics about the graph structure
        
        Args:
            graph_data: Graph data container
            
        Returns:
            Dictionary with graph statistics
        """
        try:
            adjacency = self.loaders['correlation'].get_adjacency_matrix(graph_data)
            
            # Basic graph statistics
            stats = {
                'num_nodes': graph_data.num_nodes,
                'num_edges': graph_data.num_edges,
                'density': graph_data.num_edges / (graph_data.num_nodes * (graph_data.num_nodes - 1) / 2) if graph_data.num_nodes > 1 else 0,
                'node_types': list(set(node.node_type for node in graph_data.nodes)),
                'edge_types': list(set(edge.edge_type for edge in graph_data.edges))
            }
            
            # Degree statistics
            degrees = np.sum(adjacency > 0, axis=1)
            stats['degree_stats'] = {
                'mean': float(np.mean(degrees)),
                'std': float(np.std(degrees)),
                'min': int(np.min(degrees)),
                'max': int(np.max(degrees))
            }
            
            # Weight statistics (for weighted graphs)
            if graph_data.edges:
                weights = [abs(edge.weight) for edge in graph_data.edges]
                stats['weight_stats'] = {
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights))
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing graph statistics: {e}")
            return {"error": str(e)}
    
    def export_graph_data(self, graph_data: GraphData, filepath: str) -> bool:
        """
        Export graph data to JSON format for external processing
        
        Args:
            graph_data: Graph data to export
            filepath: Path to save JSON file
            
        Returns:
            True if export successful
        """
        try:
            # Prepare data for JSON serialization
            export_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_nodes': graph_data.num_nodes,
                    'num_edges': graph_data.num_edges
                },
                'nodes': [
                    {
                        'symbol': node.symbol,
                        'node_type': node.node_type,
                        'features': node.features,
                        'metadata': node.metadata
                    }
                    for node in graph_data.nodes
                ],
                'edges': [
                    {
                        'source': edge.source,
                        'target': edge.target,
                        'weight': edge.weight,
                        'edge_type': edge.edge_type,
                        'metadata': edge.metadata
                    }
                    for edge in graph_data.edges
                ]
            }
            
            # Save to file
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Graph data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting graph data: {e}")
            return False
    
    def clear_cache(self):
        """Clear the graph data cache"""
        self.graph_cache.clear()
        logger.info("GNN data cache cleared")


# Global instance
gnn_data_manager = GNNDataManager()


# Convenience functions
def load_correlation_graph(symbols: List[str], 
                          price_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[GraphData]:
    """
    Load correlation-based financial graph
    
    Args:
        symbols: List of stock symbols
        price_data: Optional price data for correlation calculation
        
    Returns:
        GraphData object or None
    """
    return gnn_data_manager.load_financial_graph(symbols, 'correlation', price_data=price_data)


def load_sector_graph(symbols: List[str], 
                     sector_mapping: Optional[Dict[str, str]] = None) -> Optional[GraphData]:
    """
    Load sector-based financial graph
    
    Args:
        symbols: List of stock symbols
        sector_mapping: Optional mapping of symbols to sectors
        
    Returns:
        GraphData object or None
    """
    loader = SectorGraphLoader(sector_mapping)
    return gnn_data_manager.load_financial_graph(symbols, 'sector')


def get_adjacency_matrix(graph_data: GraphData, graph_type: str = 'correlation') -> Optional[np.ndarray]:
    """
    Get adjacency matrix from graph data
    
    Args:
        graph_data: Graph data container
        graph_type: Type of graph for appropriate processing
        
    Returns:
        Adjacency matrix or None
    """
    if graph_type in gnn_data_manager.loaders:
        return gnn_data_manager.loaders[graph_type].get_adjacency_matrix(graph_data)
    return None


def export_graph_for_gnn(graph_data: GraphData, 
                        filepath: str = "data/financial_graph.json") -> bool:
    """
    Export graph data for external GNN frameworks
    
    Args:
        graph_data: Graph data to export
        filepath: Path to save exported data
        
    Returns:
        True if export successful
    """
    return gnn_data_manager.export_graph_data(graph_data, filepath)