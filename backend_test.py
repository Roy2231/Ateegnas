import requests
import sys
import json
import time
from datetime import datetime
import io

class Ateegnas50xScalingTester:
    def __init__(self, base_url="https://parallel-text-anno.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.project_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}"
        headers = {'Content-Type': 'application/json'} if not files else {}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files)
                else:
                    response = requests.post(url, json=data, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_api_health(self):
        """Test API health check"""
        success, response = self.run_test(
            "API Health Check",
            "GET",
            "",
            200
        )
        return success

    def test_create_project_with_3_agents(self):
        """Create a project with 3 agents (150 total instances: 50×3)"""
        project_data = {
            "name": f"50x Scaling Test Project {datetime.now().strftime('%H%M%S')}",
            "description": "Testing 50x agent scaling with 3 agent types",
            "ontology": "Extract sentiment, entities, and classify text for customer feedback analysis",
            "selected_agents": ["sentiment_analysis", "entity_extraction", "text_classification"]
        }
        
        success, response = self.run_test(
            "Create Project with 3 Agents (150 total instances)",
            "POST",
            "projects",
            200,
            data=project_data
        )
        
        if success and 'id' in response:
            self.project_id = response['id']
            print(f"📝 Project created with ID: {self.project_id}")
            print(f"🚀 Expected total agents: {len(project_data['selected_agents']) * 50} = 150 agents")
            return True
        return False

    def test_upload_test_data(self):
        """Upload test CSV data"""
        if not self.project_id:
            print("❌ No project ID available for upload test")
            return False

        # Create test CSV content as specified in the request
        csv_content = '''text
"Amazing product quality, fast shipping, excellent customer service!"
"Disappointing experience, slow delivery, poor communication."
"Good value for money, average quality, could be improved."'''
        
        files = {
            'file': ('test_data.csv', io.StringIO(csv_content), 'text/csv')
        }
        
        success, response = self.run_test(
            "Upload Test Dataset",
            "POST",
            f"projects/{self.project_id}/upload",
            200,
            files=files
        )
        
        if success:
            print(f"📊 Uploaded {response.get('records_count', 0)} records")
            print(f"🔄 Processing status: {response.get('status', 'unknown')}")
            return True
        return False

    def test_project_status_and_scaling(self):
        """Check project status and verify 50x scaling information"""
        if not self.project_id:
            print("❌ No project ID available for status test")
            return False

        success, response = self.run_test(
            "Get Project Status and Scaling Info",
            "GET",
            f"projects/{self.project_id}",
            200
        )
        
        if success:
            print(f"📊 Project Status: {response.get('status', 'unknown')}")
            print(f"📈 Total Records: {response.get('total_records', 0)}")
            print(f"✅ Processed Records: {response.get('processed_records', 0)}")
            print(f"🤖 Selected Agents: {response.get('selected_agents', [])}")
            
            # Calculate expected total agents
            agent_count = len(response.get('selected_agents', []))
            expected_total_agents = agent_count * 50
            print(f"🚀 Expected Total Agent Instances: {expected_total_agents}")
            
            # Check for scaling metadata if processing is complete
            if 'total_agents_used' in response:
                print(f"🎯 Actual Total Agents Used: {response['total_agents_used']}")
                print(f"⚡ Processing Scale: {response.get('processing_scale', 'N/A')}")
            
            return True
        return False

    def wait_for_processing_completion(self, max_wait_time=300):
        """Wait for background processing to complete"""
        if not self.project_id:
            print("❌ No project ID available for processing wait")
            return False

        print(f"⏳ Waiting for 50x parallel processing to complete (max {max_wait_time}s)...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            success, response = self.run_test(
                "Check Processing Progress",
                "GET",
                f"projects/{self.project_id}",
                200
            )
            
            if success:
                status = response.get('status', 'unknown')
                processed = response.get('processed_records', 0)
                total = response.get('total_records', 0)
                
                print(f"🔄 Status: {status}, Progress: {processed}/{total}")
                
                if status == 'completed':
                    print("✅ Processing completed successfully!")
                    return True
                elif status == 'failed':
                    print("❌ Processing failed!")
                    return False
                
            time.sleep(10)  # Wait 10 seconds between checks
        
        print("⏰ Timeout waiting for processing completion")
        return False

    def test_consensus_results(self):
        """Test consensus results from 50 instances"""
        if not self.project_id:
            print("❌ No project ID available for results test")
            return False

        success, response = self.run_test(
            "Get Consensus Results from 50x Instances",
            "GET",
            f"projects/{self.project_id}/results",
            200
        )
        
        if success and isinstance(response, list) and len(response) > 0:
            print(f"📊 Retrieved {len(response)} annotation results")
            
            # Analyze first result for 50x scaling evidence
            first_result = response[0]
            agent_results = first_result.get('agent_results', {})
            
            print(f"🔍 Analyzing consensus results structure:")
            for agent_type, agent_data in agent_results.items():
                print(f"  🤖 Agent Type: {agent_type}")
                
                if isinstance(agent_data, dict):
                    instance_count = agent_data.get('instance_count', 0)
                    consensus_result = agent_data.get('consensus_result', {})
                    aggregated_results = agent_data.get('aggregated_results', [])
                    
                    print(f"    📈 Instance Count: {instance_count}")
                    print(f"    🎯 Has Consensus Result: {bool(consensus_result)}")
                    print(f"    📋 Aggregated Results Count: {len(aggregated_results)}")
                    
                    # Check for 50x scaling indicators
                    if consensus_result and isinstance(consensus_result, dict):
                        consensus_from = consensus_result.get('consensus_from_instances', 0)
                        processing_scale = consensus_result.get('processing_scale', '')
                        print(f"    🚀 Consensus From Instances: {consensus_from}")
                        print(f"    ⚡ Processing Scale: {processing_scale}")
                        
                        if consensus_from == 50 or '50x' in processing_scale:
                            print(f"    ✅ 50x scaling confirmed for {agent_type}")
                        else:
                            print(f"    ⚠️  50x scaling not confirmed for {agent_type}")
            
            return True
        else:
            print("❌ No results found or invalid response format")
            return False

    def test_export_functionality(self):
        """Test export functionality with 50x scale metadata"""
        if not self.project_id:
            print("❌ No project ID available for export test")
            return False

        success, response = self.run_test(
            "Export Project with 50x Scale Metadata",
            "GET",
            f"projects/{self.project_id}/export",
            200
        )
        
        if success:
            print(f"📦 Export successful")
            
            # Analyze export structure
            if isinstance(response, dict):
                project_data = response.get('project', {})
                annotations = response.get('annotations', [])
                
                print(f"📊 Project Data Keys: {list(project_data.keys())}")
                print(f"📋 Annotations Count: {len(annotations)}")
                
                # Check for scaling metadata in project
                if 'total_agents_used' in project_data:
                    print(f"🚀 Total Agents Used: {project_data['total_agents_used']}")
                if 'processing_scale' in project_data:
                    print(f"⚡ Processing Scale: {project_data['processing_scale']}")
                
                # Check annotation structure
                if annotations and len(annotations) > 0:
                    first_annotation = annotations[0]
                    agent_results = first_annotation.get('agent_results', {})
                    print(f"🔍 First annotation has {len(agent_results)} agent result types")
                
                return True
            else:
                print("⚠️  Export response format unexpected")
                return False
        return False

    def test_cleanup(self):
        """Clean up test project"""
        if not self.project_id:
            print("❌ No project ID available for cleanup")
            return True

        success, response = self.run_test(
            "Delete Test Project",
            "DELETE",
            f"projects/{self.project_id}",
            200
        )
        
        if success:
            print(f"🗑️  Test project {self.project_id} deleted successfully")
            return True
        return False

def main():
    print("🚀 Starting Ateegnas 50x Agent Scaling Tests")
    print("=" * 60)
    
    tester = Ateegnas50xScalingTester()
    
    # Test sequence
    tests = [
        ("API Health Check", tester.test_api_health),
        ("Create Project with 3 Agents", tester.test_create_project_with_3_agents),
        ("Upload Test Data", tester.test_upload_test_data),
        ("Check Project Status", tester.test_project_status_and_scaling),
        ("Wait for Processing", tester.wait_for_processing_completion),
        ("Verify Consensus Results", tester.test_consensus_results),
        ("Test Export Functionality", tester.test_export_functionality),
        ("Cleanup", tester.test_cleanup)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            if not result:
                print(f"❌ {test_name} failed - stopping tests")
                break
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            break
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"📊 Final Test Results:")
    print(f"✅ Tests Passed: {tester.tests_passed}")
    print(f"📊 Tests Run: {tester.tests_run}")
    print(f"📈 Success Rate: {(tester.tests_passed/tester.tests_run*100):.1f}%" if tester.tests_run > 0 else "0%")
    
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())