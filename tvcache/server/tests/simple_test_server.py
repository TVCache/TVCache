"""Simple test script for TVCache server without pytest.

This test suite validates the TVCache server HTTP API endpoints including
storage, exact match retrieval, and prefix match functionality. The tests
require a running instance of the TVCache server.

Usage:
    1. Start the server: python tvcache_server.py
    2. Run tests: python tests/simple_test_server.py
"""

import requests
import sys

# Configuration
BASE_URL = 'http://localhost:8000'


class TVCacheTestClient:
    """Client for making HTTP requests to TVCache server during tests."""

    def __init__(self, base_url: str):
        """Initialize the test client.

        Args:
            base_url: Base URL of the TVCache server
        """
        self.base_url = base_url

    def put(self, task_name: str, history: list, env_id: str, value: str):
        """Store tool call history.

        Args:
            task_name: The name of the task
            history: List of tool calls
            env_id: Environment ID
            value: Value to store

        Returns:
            Response object
        """
        data = {
            "task_name": task_name,
            "history": history,
            "env_id": env_id,
            "value": value
        }
        return requests.put(f'{self.base_url}/put', json=data)

    def get(self, task_name: str, tool_calls: list):
        """Get cached environment by exact match.

        Args:
            task_name: The name of the task
            tool_calls: List of tool calls to match

        Returns:
            Response object
        """
        return requests.get(
            f'{self.base_url}/get',
            params={'task_name': task_name, 'tool_calls': tool_calls}
        )

    def prefix_match(self, task_name: str, tool_calls: list):
        """Find cached environment by prefix match.

        Args:
            task_name: The name of the task
            tool_calls: List of tool calls to match

        Returns:
            Response object
        """
        prefix_data = {
            "task_name": task_name,
            "tool_calls": tool_calls
        }
        return requests.post(f'{self.base_url}/prefix_match', json=prefix_data)

    def check_server(self):
        """Check if the server is running.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(
                f'{self.base_url}/get',
                params={'task_name': 'test'},
                timeout=2
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


def test_put_endpoint(client):
    """Test the PUT endpoint for storing tool call history."""
    print("\n=== Testing PUT endpoint ===")

    response = client.put(
        task_name="test_task",
        history=["tool1", "tool2", "tool3"],
        env_id="env_123",
        value="test_value"
    )

    if response.status_code == 200 and response.json()['success']:
        print("✓ PUT endpoint works correctly")
        return True
    else:
        print(f"✗ PUT endpoint failed: {response.status_code} - {response.text}")
        return False


def test_get_endpoint_exact_match(client):
    """Test the GET endpoint for exact match."""
    print("\n=== Testing GET endpoint (exact match) ===")

    # First, put some data
    client.put(
        task_name="get_test_task",
        history=["tool1", "tool2"],
        env_id="env_456",
        value="test_value"
    )

    # Now get it back
    response = client.get(
        task_name='get_test_task',
        tool_calls=['tool1', 'tool2']
    )

    result = response.json()
    if (response.status_code == 200 and
        result['found'] == True and
        result['env_id'] == "env_456" and
        result['value'] == "test_value"):
        print("✓ GET endpoint works correctly")
        print(f"  Found: {result['found']}")
        print(f"  Env ID: {result['env_id']}")
        print(f"  Value: {result['value']}")
        return True
    else:
        print(f"✗ GET endpoint failed: {response.status_code}")
        print(f"  Response: {result}")
        return False


def test_get_endpoint_not_found(client):
    """Test the GET endpoint when no match is found."""
    print("\n=== Testing GET endpoint (not found) ===")

    response = client.get(
        task_name='nonexistent_task',
        tool_calls=['tool1']
    )

    result = response.json()
    if response.status_code == 200 and result['found'] == False:
        print("✓ GET endpoint correctly returns not found")
        return True
    else:
        print(f"✗ GET endpoint failed to handle not found case")
        print(f"  Response: {result}")
        return False


def test_prefix_match_endpoint(client):
    """Test the prefix match endpoint."""
    print("\n=== Testing PREFIX_MATCH endpoint ===")

    # Put some data
    client.put(
        task_name="prefix_test_task",
        history=["tool1", "tool2", "tool3"],
        env_id="env_789",
        value="test_value"
    )

    # Test prefix match with extended history
    response = client.prefix_match(
        task_name="prefix_test_task",
        tool_calls=["tool1", "tool2", "tool3", "tool4", "tool5"]
    )

    result = response.json()
    print('Prefix match response: {}'.format(result))

    if (response.status_code == 200 and
        result['found'] == True and
        result['history'] == ["tool1", "tool2", "tool3"] and
        result['env_id'] == "env_789"):
        print("✓ PREFIX_MATCH endpoint works correctly")
        print(f"  Found: {result['found']}")
        print(f"  History: {result['history']}")
        print(f"  Env ID: {result['env_id']}")
        return True
    else:
        print(f"✗ PREFIX_MATCH endpoint failed: {response.status_code}")
        print(f"  Response: {result}")
        return False


def test_prefix_match_no_match(client):
    """Test prefix match when no prefix matches."""
    print("\n=== Testing PREFIX_MATCH endpoint (no match) ===")

    # Put some data
    client.put(
        task_name="prefix_no_match_task",
        history=["tool1", "tool2"],
        env_id="env_111",
        value="test_value"
    )

    # Try to match with different tools
    response = client.prefix_match(
        task_name="prefix_no_match_task",
        tool_calls=["tool3", "tool4"]
    )

    result = response.json()
    if response.status_code == 200 and result['found'] == False:
        print("✓ PREFIX_MATCH endpoint correctly returns no match")
        return True
    else:
        print(f"✗ PREFIX_MATCH endpoint failed to handle no match case")
        print(f"  Response: {result}")
        return False

def test_multiple_updates(client):
    task_name = "multi-update"
    env_id = "env-098"
    
    client.put(
        task_name=task_name,
        history=["tool1", "tool2", "tool3"],
        env_id=env_id,
        value="test_value_3"
    )

    response = client.prefix_match(
        task_name=task_name,
        tool_calls=["tool1", "tool2", "toolx"]
    )

    result = response.json()
    print('Multi-match response for {} {}'.format(["tool1", "tool2", "toolx"], result))

    # Test prefix match with extended history
    response = client.prefix_match(
        task_name=task_name,
        tool_calls=["tool1", "tool2", "tool3", "tool4", "toolx"]
    )

    result = response.json()
    print('Multi-match response: {}'.format(result))

    client.put(
        task_name=task_name,
        history=["tool1", "tool2", "tool3", "tool4"],
        env_id=env_id,
        value="test_value4"
    )

    req_list = ["tool1", "tool2", "tool3", "tool4"]

    response = client.get(
        task_name=task_name,
        tool_calls=req_list
    )

    result = response.json()
    print(f'Value of latest get for {req_list}: {result}')

    req_list = ["tool1", "tool2", "tool3"]

    response = client.get(
        task_name=task_name,
        tool_calls=req_list
    )

    result = response.json()
    print(f'Value of latest get for {req_list}: {result}')

    response = client.prefix_match(
        task_name=task_name,
        tool_calls=["tool1", "tool2", "tool3", "tool4", "tool5"]
    )

    result = response.json()
    print('Multi-match response: {}'.format(result))

    return True

def test_multiple_tasks(client):
    """Test that multiple tasks can be stored independently."""
    print("\n=== Testing multiple independent tasks ===")

    # Put data for task 1
    client.put(
        task_name="multi_task1",
        history=["tool1", "tool2"],
        env_id="env_task1",
        value="value1"
    )

    # Put data for task 2
    client.put(
        task_name="multi_task2",
        history=["tool3", "tool4"],
        env_id="env_task2",
        value="value2"
    )

    # Verify task 1
    response1 = client.get(
        task_name='multi_task1',
        tool_calls=['tool1', 'tool2']
    )
    result1 = response1.json()

    # Verify task 2
    response2 = client.get(
        task_name='multi_task2',
        tool_calls=['tool3', 'tool4']
    )
    result2 = response2.json()

    if (result1['found'] and result1['env_id'] == "env_task1" and result1['value'] == "value1" and
        result2['found'] and result2['env_id'] == "env_task2" and result2['value'] == "value2"):
        print("✓ Multiple tasks stored and retrieved independently")
        print(f"  Task 1: {result1['env_id']} - {result1['value']}")
        print(f"  Task 2: {result2['env_id']} - {result2['value']}")
        return True
    else:
        print(f"✗ Multiple tasks test failed")
        print(f"  Task 1: {result1}")
        print(f"  Task 2: {result2}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TVCache Server Simple Test Suite")
    print("=" * 60)

    # Create client
    client = TVCacheTestClient(BASE_URL)

    # Check if server is running
    print(f"Checking if server is running at {BASE_URL}...")
    if client.check_server():
        print(f"✓ Server is running")
    else:
        print(f"✗ Server is not accessible")
        print(f"\nPlease start the server first:")
        print(f"  python -m tvclient.utils.tvcache_server")
        sys.exit(1)

    # Run all tests
    tests = [
        # test_put_endpoint,
        # test_get_endpoint_exact_match,
        # test_get_endpoint_not_found,
        # test_prefix_match_endpoint,
        # test_prefix_match_no_match,
        # test_multiple_tasks,
        test_multiple_updates
    ]

    results = []
    for test in tests:
        try:
            results.append(test(client))
        except Exception as e:
            print(f"✗ Test {test.__name__} raised an exception: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
