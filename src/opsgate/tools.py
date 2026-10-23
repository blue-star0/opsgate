"""EC2 tools for the OpsGate agent.

Default backend is an in-memory fake fleet so the demo is safe and repeatable.
Set OPSGATE_LIVE=1 (and install boto3) to hit real EC2 in your own account.
Live mode only ever calls the three EC2 APIs below.
"""

from __future__ import annotations

import json
import os

from strands import tool

LIVE = os.environ.get("OPSGATE_LIVE") == "1"

_FLEET = {
    "i-0a1b2c3d4e5f60001": {"name": "web-prod-1", "state": "running", "type": "t3.medium"},
    "i-0a1b2c3d4e5f60002": {"name": "web-prod-2", "state": "running", "type": "t3.medium"},
    "i-0a1b2c3d4e5f60003": {"name": "db-prod", "state": "running", "type": "r5.large"},
    "i-0a1b2c3d4e5f60004": {"name": "dev-sandbox", "state": "running", "type": "t3.small"},
    "i-0a1b2c3d4e5f60005": {"name": "old-batch-runner", "state": "stopped", "type": "c5.xlarge"},
}


def _ec2():
    import boto3  # imported lazily so the fake mode needs no AWS libraries

    return boto3.client("ec2", region_name=os.environ.get("AWS_REGION", "us-east-1"))


@tool
def list_instances() -> str:
    """List EC2 instances with their id, name, state and type."""
    if LIVE:
        out = []
        for r in _ec2().describe_instances()["Reservations"]:
            for i in r["Instances"]:
                name = next((t["Value"] for t in i.get("Tags", []) if t["Key"] == "Name"), "")
                out.append(
                    {"id": i["InstanceId"], "name": name, "state": i["State"]["Name"], "type": i["InstanceType"]}
                )
        return json.dumps(out)
    return json.dumps([{"id": k, **v} for k, v in _FLEET.items()])


@tool
def stop_instance(instance_id: str) -> str:
    """Stop (not terminate) one EC2 instance. Needs the exact instance id."""
    if LIVE:
        _ec2().stop_instances(InstanceIds=[instance_id])
        return f"stopping {instance_id}"
    if instance_id not in _FLEET:
        return f"error: no such instance {instance_id}"
    _FLEET[instance_id]["state"] = "stopped"
    return f"{instance_id} ({_FLEET[instance_id]['name']}) is now stopped"


@tool
def terminate_instance(instance_id: str) -> str:
    """Permanently terminate one EC2 instance. Cannot be undone. Needs the exact instance id."""
    if LIVE:
        _ec2().terminate_instances(InstanceIds=[instance_id])
        return f"terminating {instance_id}"
    if instance_id not in _FLEET:
        return f"error: no such instance {instance_id}"
    name = _FLEET[instance_id]["name"]
    _FLEET[instance_id]["state"] = "terminated"
    return f"{instance_id} ({name}) terminated"


ALL_TOOLS = [list_instances, stop_instance, terminate_instance]
# Read-only tool: nothing to vet, so skip the Decider round trip.
READ_ONLY = {"list_instances"}
