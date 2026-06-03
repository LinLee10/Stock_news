#!/usr/bin/env python3
"""
Integration tests for GNN scaffold module
Tests data loaders and graph construction without heavy ML dependencies
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analytics.gnn_scaffold import (
    GraphNode, GraphEdge, GraphData,
    BaseDataLoader, CorrelationGraphLoader, SectorGraphLoader,
    GNNDataManager
)
from config.feature_flags import feature_flags


class TestGNNScaffold:
    """Test suite for GNN scaffold functionality"""
    
    def setup_method(self):
        """Set up test data before each test"""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        np.random.seed(42)  # For reproducible tests
        base_returns = np.random.normal(0.001, 0.02, len(dates))
        
        # Create correlated price series
        aapl_returns = base_returns + np.random.normal(0, 0.01, len(dates))
        aapl_prices = 150 * (1 + aapl_returns).cumprod()
        
        msft_returns = 0.7 * base_returns + np.random.normal(0, 0.015, len(dates))
        msft_prices = 300 * (1 + msft_returns).cumprod()
        
        tsla_returns = 0.3 * base_returns + np.random.normal(0, 0.03, len(dates))
        tsla_prices = 200 * (1 + tsla_returns).cumprod()
        
        self.price_data = {
            'AAPL': pd.DataFrame({
                'Date': dates,
                'Close': aapl_prices,
                'Volume': np.random.randint(10000000, 100000000, len(dates))
            }),
            'MSFT': pd.DataFrame({
                'Date': dates, 
                'Close': msft_prices,
                'Volume': np.random.randint(5000000, 80000000, len(dates))
            }),
            'TSLA': pd.DataFrame({
                'Date': dates,
                'Close': tsla_prices,
                'Volume': np.random.randint(20000000, 150000000, len(dates))
            })
        }
        
        # Sample sector data
        self.sector_data = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'TSLA': 'Automotive'
        }
        
        # Enable GNN scaffold for testing
        feature_flags.set_flag('enable_gnn_scaffold', True)
    
    def teardown_method(self):
        """Clean up after each test"""
        feature_flags.set_flag('enable_gnn_scaffold', False)


class TestGraphDataStructures:
    """Test basic graph data structures"""
    
    def test_graph_node_creation(self):
        """Test GraphNode creation and attributes"""
        features = {'price': 150.0, 'volume': 1000000, 'volatility': 0.02}
        node = GraphNode(symbol='AAPL', features=features)
        
        assert node.symbol == 'AAPL'
        assert node.features == features
        assert node.features['price'] == 150.0
        assert len(node.features) == 3
    
    def test_graph_edge_creation(self):
        """Test GraphEdge creation and attributes"""
        edge = GraphEdge(
            source='AAPL',
            target='MSFT', 
            weight=0.75,
            edge_type='correlation'
        )
        
        assert edge.source == 'AAPL'
        assert edge.target == 'MSFT'
        assert edge.weight == 0.75
        assert edge.edge_type == 'correlation'
    
    def test_graph_data_creation(self):
        """Test GraphData creation and manipulation"""
        nodes = [
            GraphNode('AAPL', {'price': 150.0}),
            GraphNode('MSFT', {'price': 300.0})
        ]
        edges = [
            GraphEdge('AAPL', 'MSFT', 0.8, 'correlation')
        ]
        
        graph_data = GraphData(nodes=nodes, edges=edges)
        
        assert len(graph_data.nodes) == 2
        assert len(graph_data.edges) == 1
        assert graph_data.get_node('AAPL') is not None
        assert graph_data.get_node('AAPL').symbol == 'AAPL'
        assert graph_data.get_node('NONEXISTENT') is None
    
    def test_graph_data_adjacency_matrix(self):
        """Test adjacency matrix generation"""
        nodes = [
            GraphNode('AAPL', {'price': 150.0}),
            GraphNode('MSFT', {'price': 300.0}),
            GraphNode('TSLA', {'price': 200.0})
        ]
        edges = [
            GraphEdge('AAPL', 'MSFT', 0.8, 'correlation'),
            GraphEdge('MSFT', 'TSLA', 0.3, 'correlation'),
            GraphEdge('AAPL', 'TSLA', 0.2, 'correlation')
        ]
        
        graph_data = GraphData(nodes=nodes, edges=edges)
        adj_matrix = graph_data.get_adjacency_matrix()
        
        assert adj_matrix.shape == (3, 3)
        assert isinstance(adj_matrix, np.ndarray)
        
        # Check specific edge weights
        symbol_to_index = {node.symbol: i for i, node in enumerate(nodes)}
        aapl_idx = symbol_to_index['AAPL']
        msft_idx = symbol_to_index['MSFT']
        tsla_idx = symbol_to_index['TSLA']
        
        assert adj_matrix[aapl_idx, msft_idx] == 0.8
        assert adj_matrix[msft_idx, tsla_idx] == 0.3
        assert adj_matrix[aapl_idx, tsla_idx] == 0.2
        
        # Check symmetry (undirected graph)
        assert adj_matrix[msft_idx, aapl_idx] == 0.8
        assert adj_matrix[tsla_idx, msft_idx] == 0.3
        assert adj_matrix[tsla_idx, aapl_idx] == 0.2
    
    def test_graph_data_feature_matrix(self):
        """Test feature matrix generation"""
        nodes = [
            GraphNode('AAPL', {'price': 150.0, 'volume': 1000000}),
            GraphNode('MSFT', {'price': 300.0, 'volume': 500000}),
            GraphNode('TSLA', {'price': 200.0, 'volume': 2000000})
        ]
        
        graph_data = GraphData(nodes=nodes, edges=[])
        
        # Test with specific features
        feature_matrix = graph_data.get_feature_matrix(['price', 'volume'])
        
        assert feature_matrix.shape == (3, 2)
        assert feature_matrix[0, 0] == 150.0  # AAPL price
        assert feature_matrix[0, 1] == 1000000  # AAPL volume
        assert feature_matrix[1, 0] == 300.0  # MSFT price
        
        # Test with all features
        all_features_matrix = graph_data.get_feature_matrix()
        assert all_features_matrix.shape == (3, 2)  # 2 features per node


class TestCorrelationGraphLoader:
    """Test correlation-based graph loader"""
    
    def setup_method(self):
        """Set up test data"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create correlated data
        base_returns = np.random.normal(0.001, 0.02, len(dates))
        aapl_returns = base_returns + np.random.normal(0, 0.01, len(dates))
        msft_returns = 0.7 * base_returns + np.random.normal(0, 0.015, len(dates))
        tsla_returns = 0.3 * base_returns + np.random.normal(0, 0.03, len(dates))
        
        self.price_data = {
            'AAPL': pd.DataFrame({
                'Date': dates,
                'Close': 150 * (1 + aapl_returns).cumprod(),
                'Volume': np.random.randint(10000000, 100000000, len(dates))
            }),
            'MSFT': pd.DataFrame({
                'Date': dates,
                'Close': 300 * (1 + msft_returns).cumprod(),
                'Volume': np.random.randint(5000000, 80000000, len(dates))
            }),
            'TSLA': pd.DataFrame({
                'Date': dates,
                'Close': 200 * (1 + tsla_returns).cumprod(),
                'Volume': np.random.randint(20000000, 150000000, len(dates))
            })
        }
        
        feature_flags.set_flag('enable_gnn_scaffold', True)
        feature_flags.set_flag('enable_correlation', True)
    
    def teardown_method(self):
        """Clean up"""
        feature_flags.set_flag('enable_gnn_scaffold', False)
        feature_flags.set_flag('enable_correlation', False)
    
    def test_correlation_graph_loading(self):
        """Test correlation graph data loading"""
        loader = CorrelationGraphLoader(correlation_threshold=0.1)
        symbols = ['AAPL', 'MSFT', 'TSLA']
        
        graph_data = loader.load_graph_data(symbols, price_data=self.price_data)
        
        assert graph_data is not None
        assert len(graph_data.nodes) == 3
        
        # Check nodes have expected symbols
        node_symbols = {node.symbol for node in graph_data.nodes}
        assert node_symbols == {'AAPL', 'MSFT', 'TSLA'}
        
        # Check nodes have features
        for node in graph_data.nodes:
            assert 'avg_price' in node.features
            assert 'avg_volume' in node.features
            assert 'volatility' in node.features
            assert node.features['avg_price'] > 0
            assert node.features['avg_volume'] > 0
        
        # Check edges exist (should have correlations above threshold)
        assert len(graph_data.edges) > 0
        
        for edge in graph_data.edges:
            assert edge.edge_type == 'correlation'
            assert abs(edge.weight) >= 0.1  # Above threshold
            assert edge.source in symbols
            assert edge.target in symbols
            assert edge.source != edge.target  # No self-loops
    
    def test_correlation_threshold_filtering(self):
        """Test correlation threshold filtering"""
        # High threshold - should have fewer edges
        loader_high = CorrelationGraphLoader(correlation_threshold=0.9)
        graph_high = loader_high.load_graph_data(
            ['AAPL', 'MSFT', 'TSLA'], 
            price_data=self.price_data
        )
        
        # Low threshold - should have more edges
        loader_low = CorrelationGraphLoader(correlation_threshold=0.1)
        graph_low = loader_low.load_graph_data(
            ['AAPL', 'MSFT', 'TSLA'],
            price_data=self.price_data
        )
        
        assert graph_high is not None
        assert graph_low is not None
        
        # Low threshold should have more or equal edges
        assert len(graph_low.edges) >= len(graph_high.edges)
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        # Create minimal data
        minimal_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=5),
                'Close': [150, 151, 149, 152, 148],
                'Volume': [1000000] * 5
            })
        }
        
        loader = CorrelationGraphLoader()
        graph_data = loader.load_graph_data(['AAPL'], price_data=minimal_data)
        
        # Should return None or empty graph due to insufficient data
        assert graph_data is None or len(graph_data.nodes) == 0


class TestSectorGraphLoader:
    """Test sector-based graph loader"""
    
    def setup_method(self):
        """Set up test data"""
        self.sector_data = {
            'AAPL': 'Technology',
            'MSFT': 'Technology', 
            'GOOGL': 'Technology',
            'TSLA': 'Automotive',
            'F': 'Automotive',
            'JPM': 'Financial'
        }
        
        # Create sample price data for feature calculation
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.price_data = {}
        
        for symbol in self.sector_data.keys():
            np.random.seed(ord(symbol[0]))  # Deterministic but different per symbol
            prices = 100 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod()
            self.price_data[symbol] = pd.DataFrame({
                'Date': dates,
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            })
        
        feature_flags.set_flag('enable_gnn_scaffold', True)
    
    def teardown_method(self):
        """Clean up"""
        feature_flags.set_flag('enable_gnn_scaffold', False)
    
    def test_sector_graph_loading(self):
        """Test sector-based graph construction"""
        loader = SectorGraphLoader()
        symbols = list(self.sector_data.keys())
        
        graph_data = loader.load_graph_data(
            symbols, 
            sector_data=self.sector_data,
            price_data=self.price_data
        )
        
        assert graph_data is not None
        assert len(graph_data.nodes) == 6
        
        # Check all symbols are present as nodes
        node_symbols = {node.symbol for node in graph_data.nodes}
        assert node_symbols == set(symbols)
        
        # Check edges connect symbols in the same sector
        tech_symbols = {'AAPL', 'MSFT', 'GOOGL'}
        auto_symbols = {'TSLA', 'F'}
        
        tech_edges = []
        auto_edges = []
        
        for edge in graph_data.edges:
            if edge.source in tech_symbols and edge.target in tech_symbols:
                tech_edges.append(edge)
            elif edge.source in auto_symbols and edge.target in auto_symbols:
                auto_edges.append(edge)
            
            assert edge.edge_type == 'sector'
            assert edge.weight == 1.0  # Default sector connection weight
        
        # Should have edges within Technology sector (3 nodes = 3 edges for complete subgraph)
        assert len(tech_edges) == 3  # AAPL-MSFT, AAPL-GOOGL, MSFT-GOOGL
        
        # Should have edges within Automotive sector (2 nodes = 1 edge)
        assert len(auto_edges) == 1  # TSLA-F
        
        # JPM (Financial) should have no sector connections (single node in sector)
    
    def test_sector_missing_data(self):
        """Test behavior when sector data is missing for some symbols"""
        incomplete_sectors = {
            'AAPL': 'Technology',
            'MSFT': 'Technology'
            # Missing TSLA sector info
        }
        
        loader = SectorGraphLoader()
        graph_data = loader.load_graph_data(
            ['AAPL', 'MSFT', 'TSLA'],
            sector_data=incomplete_sectors,
            price_data=self.price_data
        )
        
        assert graph_data is not None
        # Should still create nodes for all symbols
        assert len(graph_data.nodes) == 3
        
        # But only edges for symbols with sector data
        edge_symbols = set()
        for edge in graph_data.edges:
            edge_symbols.add(edge.source)
            edge_symbols.add(edge.target)
        
        # Only AAPL and MSFT should have edges
        assert edge_symbols.issubset({'AAPL', 'MSFT'})


class TestGNNDataManager:
    """Test high-level GNN data management"""
    
    def setup_method(self):
        """Set up test data"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.price_data = {
            'AAPL': pd.DataFrame({
                'Date': dates,
                'Close': 150 + np.random.randn(len(dates)) * 5,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }),
            'MSFT': pd.DataFrame({
                'Date': dates,
                'Close': 300 + np.random.randn(len(dates)) * 10,
                'Volume': np.random.randint(500000, 8000000, len(dates))
            })
        }
        
        self.sector_data = {
            'AAPL': 'Technology',
            'MSFT': 'Technology'
        }
        
        feature_flags.set_flag('enable_gnn_scaffold', True)
        feature_flags.set_flag('enable_correlation', True)
    
    def teardown_method(self):
        """Clean up"""
        feature_flags.set_flag('enable_gnn_scaffold', False)
        feature_flags.set_flag('enable_correlation', False)
    
    def test_gnn_manager_creation(self):
        """Test GNN data manager initialization"""
        manager = GNNDataManager()
        
        assert manager is not None
        assert hasattr(manager, 'correlation_loader')
        assert hasattr(manager, 'sector_loader')
    
    def test_correlation_graph_creation(self):
        """Test correlation graph creation through manager"""
        manager = GNNDataManager()
        symbols = ['AAPL', 'MSFT']
        
        graph_data = manager.create_correlation_graph(
            symbols, 
            self.price_data, 
            correlation_threshold=0.1
        )
        
        assert graph_data is not None
        assert len(graph_data.nodes) == 2
        
        # Verify graph structure
        assert graph_data.get_node('AAPL') is not None
        assert graph_data.get_node('MSFT') is not None
    
    def test_sector_graph_creation(self):
        """Test sector graph creation through manager"""
        manager = GNNDataManager()
        symbols = ['AAPL', 'MSFT']
        
        graph_data = manager.create_sector_graph(
            symbols,
            self.sector_data,
            self.price_data
        )
        
        assert graph_data is not None
        assert len(graph_data.nodes) == 2
        assert len(graph_data.edges) == 1  # Same sector connection
    
    def test_feature_flag_disabled(self):
        """Test behavior when GNN scaffold is disabled"""
        feature_flags.set_flag('enable_gnn_scaffold', False)
        
        manager = GNNDataManager()
        
        graph_data = manager.create_correlation_graph(
            ['AAPL', 'MSFT'],
            self.price_data
        )
        
        assert graph_data is None


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Set up test data"""
        feature_flags.set_flag('enable_gnn_scaffold', True)
    
    def teardown_method(self):
        """Clean up"""
        feature_flags.set_flag('enable_gnn_scaffold', False)
    
    def test_empty_symbol_list(self):
        """Test behavior with empty symbol list"""
        loader = CorrelationGraphLoader()
        graph_data = loader.load_graph_data([], price_data={})
        
        assert graph_data is None
    
    def test_missing_price_data(self):
        """Test behavior when price data is missing"""
        loader = CorrelationGraphLoader()
        graph_data = loader.load_graph_data(['AAPL'], price_data={})
        
        assert graph_data is None
    
    def test_malformed_price_data(self):
        """Test behavior with malformed price data"""
        bad_data = {
            'AAPL': pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=10),
                'BadColumn': np.random.randn(10)  # Missing required columns
            })
        }
        
        loader = CorrelationGraphLoader()
        graph_data = loader.load_graph_data(['AAPL'], price_data=bad_data)
        
        assert graph_data is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])