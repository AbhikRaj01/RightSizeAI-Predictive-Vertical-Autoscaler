import kopf
import kubernetes.client as k8s
from kubernetes import config
import requests

# Connect to your local Minikube cluster
config.load_kube_config()

# Watch Deployments that have the annotation: rightsize.ai/enabled: "true"
# Check them every 20 seconds
@kopf.on.timer('apps', 'v1', 'deployments', interval=20, annotations={'rightsize.ai/enabled': 'true'})
def rightsize_deployment(name, namespace, spec, logger, **kwargs):
    logger.info(f"Analyzing '{name}' for predictive rightsizing...")
    
    try:
        # 1. Ask the AI Brain for a prediction
        res = requests.get(f"http://127.0.0.1:8000/predict?target_deployment={name}")
        data = res.json()
        
        # 2. Check if the AI is confident enough to make a change
        if data.get('status') == 'success' and data.get('confidence') >= 0.80:
            new_cpu = data['recommended_cpu_limit']
            logger.info(f"AI confidence high ({data['confidence']}). Downshifting to {new_cpu} CPU.")
            
            # 3. Dynamically get the name of the container inside the pod
            container_name = spec['template']['spec']['containers'][0]['name']
            
            # 4. Create the precise JSON patch to alter the Kubernetes object
            patch = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": container_name,
                                "resources": {
                                    "limits": {"cpu": new_cpu}
                                }
                            }]
                        }
                    }
                }
            }
            
            # 5. Apply the patch to the live cluster
            api = k8s.AppsV1Api()
            api.patch_namespaced_deployment(name=name, namespace=namespace, body=patch)
            logger.info(f"✅ SUCCESS: Physically resized {name} to {new_cpu}!")
            
        else:
            logger.info("AI confidence too low. Skipping resize for safety.")
            
    except Exception as e:
        logger.error(f"Failed to process deployment: {e}")