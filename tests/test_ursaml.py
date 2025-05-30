import pytest
from app.ursaml.parser import parse_ursaml, serialize_ursaml

def test_parse_ursaml():
    """Test parsing UrsaML format."""
    ursaml_content = """@URSAML 0.1 sample_sdkg_01

@COLUMNS
score:[0.85,0.75,0.70,0.40]
name:["my_neural_network","mid_neural_network","random_neural_network","bad_neural_network"]
@ENDCOLUMNS

@STRUCTURE
n1->n2:1.0:"immediate"
n2->n3:2.0:"fair"
n2->n4:3.0:"distant"
@ENDSTRUCTURE

@CONTENT
n1|0.85|"my_neural_network"|{
    id:"n1"
    name:"my_neural_network"
    param:framework:"tensorflow"
    param:type:"neural_network"
    param:layers:7
    meta:score:0.85
}

n2|0.75|"mid_neural_network"|{
    id:"n2"
    name:"mid_neural_network"
    param:framework:"tensorflow"
    param:type:"neural_network"
    param:layers:5
    meta:score:0.75
}
@ENDCONTENT

@END sample_sdkg_01"""

    # Parse the content
    result = parse_ursaml(ursaml_content)
    
    # Check header
    assert result['version'] == '0.1'
    assert result['identifier'] == 'sample_sdkg_01'
    
    # Check columns
    assert 'score' in result['columns']
    assert 'name' in result['columns']
    assert len(result['column_values']['score']) == 4
    
    # Check structure (edges)
    assert len(result['structure']) == 3
    assert result['structure'][0] == ('n1', 'n2', 1.0, 'immediate')
    
    # Check nodes
    assert 'n1' in result['nodes']
    assert result['nodes']['n1']['columns']['score'] == 0.85
    assert result['nodes']['n1']['columns']['name'] == 'my_neural_network'
    assert result['nodes']['n1']['detailed']['params']['framework'] == 'tensorflow'

def test_serialize_ursaml():
    """Test serializing to UrsaML format."""
    data = {
        'version': '0.1',
        'identifier': 'test_graph',
        'columns': ['score', 'name'],
        'column_values': {
            'score': [0.9, 0.8],
            'name': ['model1', 'model2']
        },
        'structure': [
            ('n1', 'n2', 1.0, 'connected')
        ],
        'nodes': {
            'n1': {
                'columns': {'score': 0.9, 'name': 'model1'},
                'detailed': {
                    'id': 'n1',
                    'name': 'model1',
                    'params': {'framework': 'pytorch'},
                    'meta': {'trained': True}
                }
            },
            'n2': {
                'columns': {'score': 0.8, 'name': 'model2'},
                'detailed': {
                    'id': 'n2',
                    'name': 'model2',
                    'params': {'framework': 'tensorflow'},
                    'meta': {'trained': False}
                }
            }
        }
    }
    
    # Serialize to UrsaML
    result = serialize_ursaml(data)
    
    # Check the result contains key sections
    assert '@URSAML 0.1 test_graph' in result
    assert '@COLUMNS' in result
    assert '@STRUCTURE' in result
    assert '@CONTENT' in result
    assert 'n1->n2:1.0:"connected"' in result 