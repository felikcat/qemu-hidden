#!/usr/bin/env python3
"""
CPU Pinning Script for VFIO/QEMU/libvirt setups.

Generates optimal CPU pinning configurations for virtual machines,
taking into account CPU topology, NUMA nodes, and SMT (hyperthreading).
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.dom import minidom
from xml.etree import ElementTree as ET


@dataclass
class CPUCore:
    """Represents a physical CPU core with its threads."""
    core_id: int
    physical_id: int  # Socket/NUMA node
    threads: list[int] = field(default_factory=list)
    is_p_core: bool = True  # True for P-cores (or non-hybrid), False for E-cores


@dataclass
class CPUTopology:
    """Represents the full CPU topology of the system."""
    cores: dict[tuple[int, int], CPUCore] = field(default_factory=dict)  # (socket, core) -> CPUCore
    numa_nodes: dict[int, list[int]] = field(default_factory=dict)  # node -> cpu list
    total_threads: int = 0
    sockets: int = 0
    cores_per_socket: int = 0
    threads_per_core: int = 0
    is_hybrid: bool = False  # True if Intel hybrid CPU (P+E cores)
    p_cores_count: int = 0
    e_cores_count: int = 0

    def get_core_pairs(self, socket: int = 0) -> list[tuple[int, ...]]:
        """Get list of (thread1, thread2, ...) tuples for each core on a socket.

        On hybrid CPUs (Intel 12th gen+), automatically excludes E-cores
        since they are unsuitable for VM pinning due to lower performance.

        Args:
            socket: Socket/NUMA node to filter by
        """
        pairs = []
        for (sock, _), core in sorted(self.cores.items()):
            if sock == socket:
                # Always skip E-cores on hybrid CPUs - they tank VM performance
                if self.is_hybrid and not core.is_p_core:
                    continue
                pairs.append(tuple(sorted(core.threads)))
        return pairs

    def get_all_core_pairs(self, socket: int = 0) -> list[tuple[int, ...]]:
        """Get ALL core pairs including E-cores (for host allocation only)."""
        pairs = []
        for (sock, _), core in sorted(self.cores.items()):
            if sock == socket:
                pairs.append(tuple(sorted(core.threads)))
        return pairs

    def get_all_threads(self, socket: int = 0) -> list[int]:
        """Get all thread IDs for a socket."""
        threads = []
        for (sock, _), core in self.cores.items():
            if sock == socket:
                threads.extend(core.threads)
        return sorted(threads)


def parse_cpu_topology() -> CPUTopology:
    """Parse CPU topology from /proc/cpuinfo and /sys/devices/system/cpu/."""
    topology = CPUTopology()

    # Parse /proc/cpuinfo
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        print("Error: /proc/cpuinfo not found", file=sys.stderr)
        sys.exit(1)

    current_processor = None
    current_physical_id = 0
    current_core_id = 0

    with open(cpuinfo_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("processor"):
                current_processor = int(line.split(":")[1].strip())
                topology.total_threads += 1
            elif line.startswith("physical id"):
                current_physical_id = int(line.split(":")[1].strip())
            elif line.startswith("core id"):
                current_core_id = int(line.split(":")[1].strip())
            elif line == "" and current_processor is not None:
                # End of processor block
                key = (current_physical_id, current_core_id)
                if key not in topology.cores:
                    topology.cores[key] = CPUCore(
                        core_id=current_core_id,
                        physical_id=current_physical_id
                    )
                topology.cores[key].threads.append(current_processor)
                current_processor = None

    # Handle last processor if file doesn't end with blank line
    if current_processor is not None:
        key = (current_physical_id, current_core_id)
        if key not in topology.cores:
            topology.cores[key] = CPUCore(
                core_id=current_core_id,
                physical_id=current_physical_id
            )
        topology.cores[key].threads.append(current_processor)

    # Calculate topology stats
    sockets = set()
    cores_per_socket = {}
    for (socket, _), core in topology.cores.items():
        sockets.add(socket)
        cores_per_socket[socket] = cores_per_socket.get(socket, 0) + 1

    topology.sockets = len(sockets)
    topology.cores_per_socket = max(cores_per_socket.values()) if cores_per_socket else 0
    topology.threads_per_core = max(len(c.threads) for c in topology.cores.values()) if topology.cores else 1

    # Parse NUMA topology
    numa_path = Path("/sys/devices/system/node")
    if numa_path.exists():
        for node_dir in numa_path.iterdir():
            if node_dir.name.startswith("node") and node_dir.name[4:].isdigit():
                node_id = int(node_dir.name[4:])
                cpulist_path = node_dir / "cpulist"
                if cpulist_path.exists():
                    topology.numa_nodes[node_id] = parse_cpu_list(cpulist_path.read_text().strip())

    # Detect Intel hybrid CPU (P-cores vs E-cores)
    detect_hybrid_cores(topology)

    return topology


def parse_cpu_list(cpu_list: str) -> list[int]:
    """Parse a CPU list string like '0-3,8-11' into a list of integers."""
    cpus = []
    for part in cpu_list.split(","):
        if "-" in part:
            start, end = part.split("-")
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


def detect_hybrid_cores(topology: CPUTopology) -> None:
    """
    Detect Intel hybrid CPU architecture and classify P-cores vs E-cores.

    Uses cpu_capacity from sysfs (P-cores = 1024, E-cores typically ~640-680).
    Falls back to base_frequency if cpu_capacity is unavailable.

    Modifies the topology in-place, setting is_p_core on each CPUCore.
    """
    cpu_capacities: dict[int, int] = {}
    base_frequencies: dict[int, int] = {}

    # Try to read cpu_capacity for each CPU
    cpu_base_path = Path("/sys/devices/system/cpu")
    for cpu_dir in cpu_base_path.iterdir():
        if not cpu_dir.name.startswith("cpu") or not cpu_dir.name[3:].isdigit():
            continue
        cpu_id = int(cpu_dir.name[3:])

        # Method 1: cpu_capacity (most reliable on modern kernels)
        capacity_path = cpu_dir / "cpu_capacity"
        if capacity_path.exists():
            try:
                cpu_capacities[cpu_id] = int(capacity_path.read_text().strip())
            except (ValueError, OSError):
                pass

        # Method 2: base_frequency (fallback)
        freq_path = cpu_dir / "cpufreq" / "base_frequency"
        if freq_path.exists():
            try:
                base_frequencies[cpu_id] = int(freq_path.read_text().strip())
            except (ValueError, OSError):
                pass

    # Determine if this is a hybrid CPU
    capacities = set(cpu_capacities.values())
    frequencies = set(base_frequencies.values())

    # Hybrid CPUs have different capacity/frequency values for P and E cores
    is_hybrid = len(capacities) > 1 or (len(capacities) == 0 and len(frequencies) > 1)

    if not is_hybrid:
        # Not a hybrid CPU, all cores are "P-cores" (or equivalent)
        return

    topology.is_hybrid = True

    # Determine the threshold for P-cores
    if cpu_capacities:
        # P-cores have highest capacity (typically 1024)
        max_capacity = max(cpu_capacities.values())
        p_core_threshold = max_capacity * 0.9  # Allow some tolerance

        # Build set of P-core thread IDs
        p_core_threads = {cpu_id for cpu_id, cap in cpu_capacities.items() if cap >= p_core_threshold}
    elif base_frequencies:
        # P-cores have highest base frequency
        max_freq = max(base_frequencies.values())
        p_core_threshold = max_freq * 0.9

        p_core_threads = {cpu_id for cpu_id, freq in base_frequencies.items() if freq >= p_core_threshold}
    else:
        # Cannot determine, assume all are P-cores
        return

    # Update cores with P-core/E-core classification
    p_cores = 0
    e_cores = 0
    for core in topology.cores.values():
        # A core is a P-core if any of its threads are P-core threads
        core.is_p_core = any(t in p_core_threads for t in core.threads)
        if core.is_p_core:
            p_cores += 1
        else:
            e_cores += 1

    topology.p_cores_count = p_cores
    topology.e_cores_count = e_cores


def format_cpu_list(cpus: list[int]) -> str:
    """Format a list of CPUs into a compact range string."""
    if not cpus:
        return ""

    cpus = sorted(set(cpus))
    ranges = []
    start = cpus[0]
    end = cpus[0]

    for cpu in cpus[1:]:
        if cpu == end + 1:
            end = cpu
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = cpu

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ",".join(ranges)


def generate_libvirt_xml(
    topology: CPUTopology,
    vm_cores: int,
    vm_threads: int = 2,
    start_core: int = 1,
    socket: int = 0,
    emulator_pin: bool = True,
    iothreads: int = 1,
    host_reserved_cores: int = 1,
) -> str:
    """Generate libvirt XML snippet for CPU pinning configuration."""

    core_pairs = topology.get_core_pairs(socket)

    if start_core + vm_cores > len(core_pairs):
        print(f"Warning: Not enough cores. Available: {len(core_pairs)}, "
              f"Requested start: {start_core}, cores: {vm_cores}", file=sys.stderr)
        start_core = max(0, len(core_pairs) - vm_cores)

    # Select cores for VM
    selected_pairs = core_pairs[start_core:start_core + vm_cores]

    # Build XML
    root = ET.Element("domain")

    # vCPU configuration
    total_vcpus = vm_cores * vm_threads
    vcpu = ET.SubElement(root, "vcpu", placement="static")
    vcpu.text = str(total_vcpus)

    # IOThreads
    if iothreads > 0:
        iothreads_elem = ET.SubElement(root, "iothreads")
        iothreads_elem.text = str(iothreads)

    # CPU topology
    cpu = ET.SubElement(root, "cpu", mode="host-passthrough", check="none", migratable="off")
    ET.SubElement(cpu, "topology", sockets="1", dies="1", cores=str(vm_cores), threads=str(vm_threads))

    ET.SubElement(cpu, "cache", mode="passthrough")

    # Feature flags for better performance
    ET.SubElement(cpu, "feature", policy="require", name="topoext")

    # CPU tuning
    cputune = ET.SubElement(root, "cputune")

    # vCPU pinning
    vcpu_id = 0
    for pair in selected_pairs:
        for thread_idx in range(min(vm_threads, len(pair))):
            vcpupin = ET.SubElement(cputune, "vcpupin", vcpu=str(vcpu_id), cpuset=str(pair[thread_idx]))
            vcpu_id += 1

    # Emulator pinning (use host-reserved cores)
    if emulator_pin and host_reserved_cores > 0:
        host_cores = core_pairs[:host_reserved_cores]
        emulator_cpus = []
        for pair in host_cores:
            emulator_cpus.extend(pair)
        emulatorpin = ET.SubElement(cputune, "emulatorpin", cpuset=format_cpu_list(emulator_cpus))

    # IOThread pinning
    if iothreads > 0 and host_reserved_cores > 0:
        host_cores = core_pairs[:host_reserved_cores]
        iothread_cpus = []
        for pair in host_cores:
            iothread_cpus.extend(pair)
        for i in range(1, iothreads + 1):
            iothreadpin = ET.SubElement(cputune, "iothreadpin", iothread=str(i), cpuset=format_cpu_list(iothread_cpus))

    # NUMA configuration
    numatune = ET.SubElement(root, "numatune")
    ET.SubElement(numatune, "memory", mode="strict", nodeset=str(socket))

    # Format XML nicely
    xml_str = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(xml_str)

    # Extract just the relevant parts (skip domain wrapper)
    parts = []
    for child in dom.documentElement.childNodes:
        if child.nodeType == child.ELEMENT_NODE:
            parts.append(child.toprettyxml(indent="  "))

    return "\n".join(parts)


@dataclass
class CoreAllocation:
    """CPU core allocation for VM and host."""
    vm_cores: list[int]
    host_cores: list[int]
    all_cores: list[int]

    @property
    def vm_cores_str(self) -> str:
        return format_cpu_list(self.vm_cores)

    @property
    def host_cores_str(self) -> str:
        return format_cpu_list(self.host_cores)

    @property
    def all_cores_str(self) -> str:
        return format_cpu_list(self.all_cores)


def calculate_host_reserved_cores(topology: CPUTopology, vm_cores: int, socket: int = 0) -> int:
    """
    Automatically calculate how many P-cores to reserve for the host.

    On hybrid CPUs, this only counts P-cores since E-cores are always
    assigned to the host automatically.

    Heuristics:
    - Always reserve at least 1 P-core (core 0)
    - Reserve 2 P-cores if system has 8+ P-cores
    - Reserve more if needed to leave room for VM cores
    - Ensure host has at least 2 threads available

    Args:
        topology: CPU topology information
        vm_cores: Number of P-cores requested for VM
        socket: Socket/NUMA node being used

    Returns:
        Number of P-cores to reserve for host (E-cores are always for host)
    """
    # On hybrid CPUs, only count P-cores for VM allocation
    if topology.is_hybrid:
        total_physical = topology.p_cores_count
    else:
        total_physical = topology.cores_per_socket

    # Base reservation: 1 P-core for small systems, 2 for larger
    if total_physical >= 8:
        base_reserved = 2
    else:
        base_reserved = 1

    # Ensure we have enough P-cores left for the VM
    available_for_vm = total_physical - base_reserved
    if available_for_vm < vm_cores:
        # Reduce host reservation if needed, but keep at least 1 P-core
        base_reserved = max(1, total_physical - vm_cores)

    return base_reserved


def generate_core_allocation(
    topology: CPUTopology,
    vm_cores: int,
    host_reserved_cores: Optional[int] = None,
    socket: int = 0,
    start_core: Optional[int] = None,
) -> CoreAllocation:
    """
    Generate VM_CORES, HOST_CORES, and ALL_CORES based on topology.

    On hybrid CPUs (Intel 12th gen+), VM cores are selected from P-cores only.
    E-cores are automatically assigned to the host for background tasks.

    Args:
        topology: CPU topology information
        vm_cores: Number of physical cores for the VM (P-cores on hybrid)
        host_reserved_cores: Number of P-cores to reserve for host (None = auto)
        socket: Socket/NUMA node to use
        start_core: First P-core to assign to VM (None = auto, after host cores)

    Returns:
        CoreAllocation with vm_cores, host_cores, and all_cores lists
    """
    # get_core_pairs returns P-cores only on hybrid CPUs
    p_core_pairs = topology.get_core_pairs(socket)
    # get_all_core_pairs includes E-cores (for host allocation)
    all_core_pairs = topology.get_all_core_pairs(socket)

    total_p_cores = len(p_core_pairs)

    if topology.is_hybrid:
        print(f"Hybrid CPU detected: {topology.p_cores_count} P-cores, "
              f"{topology.e_cores_count} E-cores", file=sys.stderr)
        print(f"VM will use P-cores only. E-cores assigned to host.", file=sys.stderr)

    # Auto-calculate host reserved cores if not specified
    if host_reserved_cores is None:
        host_reserved_cores = calculate_host_reserved_cores(topology, vm_cores, socket)

    # Validate against P-cores (VM can only use P-cores)
    host_reserved_cores = max(1, host_reserved_cores)  # At least 1 P-core for host
    if vm_cores + host_reserved_cores > total_p_cores:
        raise ValueError(
            f"Not enough P-cores: {vm_cores} VM + {host_reserved_cores} host reserved = "
            f"{vm_cores + host_reserved_cores}, but only {total_p_cores} P-cores available"
        )

    # Auto-calculate start_core if not specified
    if start_core is None:
        start_core = host_reserved_cores

    # Ensure core 0 is never given to VM (always keep for host)
    if start_core == 0:
        start_core = 1
        print("Warning: Core 0 should be reserved for host. Adjusting start_core to 1.",
              file=sys.stderr)

    # Validate start_core
    if start_core + vm_cores > total_p_cores:
        raise ValueError(
            f"start_core={start_core} + vm_cores={vm_cores} exceeds available P-cores "
            f"({total_p_cores})"
        )

    # Build VM cores list (P-cores only)
    vm_thread_list = []
    for i in range(start_core, start_core + vm_cores):
        vm_thread_list.extend(p_core_pairs[i])

    # Build ALL cores list (includes E-cores)
    all_thread_list = []
    for pair in all_core_pairs:
        all_thread_list.extend(pair)

    # Host gets everything NOT assigned to VM (including all E-cores)
    host_thread_list = [t for t in all_thread_list if t not in vm_thread_list]

    # Ensure host has at least P-core 0's threads
    core_0_threads = p_core_pairs[0] if p_core_pairs else []
    for t in core_0_threads:
        if t not in host_thread_list:
            host_thread_list.append(t)
            if t in vm_thread_list:
                vm_thread_list.remove(t)

    return CoreAllocation(
        vm_cores=sorted(vm_thread_list),
        host_cores=sorted(host_thread_list),
        all_cores=sorted(all_thread_list),
    )


def output_shell_vars(allocation: CoreAllocation):
    """Output shell-parseable variable assignments for sourcing in bash."""
    print(f'VM_CORES="{allocation.vm_cores_str}"')
    print(f'HOST_CORES="{allocation.host_cores_str}"')
    print(f'ALL_CORES="{allocation.all_cores_str}"')


def generate_hook_script(allocation: CoreAllocation, vm_memory_mb: int) -> str:
    """Generate the libvirt qemu hook bash script."""
    return f'''#!/bin/bash

# -------------------------
# CPU topology configuration
# Auto-generated by cpu_pinning.py
# -------------------------

# VM gets these logical cores
VM_CORES="{allocation.vm_cores_str}"

# Host keeps these logical cores
HOST_CORES="{allocation.host_cores_str}"

# Total available system cores
ALL_CORES="{allocation.all_cores_str}"

# VM memory in MB (used for hugepages)
VM_MEMORY="{vm_memory_mb}"

command=$2

# -------------------------
# Function: allocate hugepages
# -------------------------
allocate_hugepages() {{
  HP_KB=$(grep Hugepagesize /proc/meminfo | awk '{{print $2}}')
  HP_MB=$((HP_KB / 1024))

  HUGEPAGES=$(( VM_MEMORY / HP_MB ))

  echo "Allocating hugepages: $HUGEPAGES pages ($VM_MEMORY MB / ${{HP_MB}}MB pagesize)"

  echo $HUGEPAGES > /proc/sys/vm/nr_hugepages

  ALLOC_PAGES=$(cat /proc/sys/vm/nr_hugepages)
  TRIES=0

  while (( ALLOC_PAGES != HUGEPAGES && TRIES < 1024 ))
  do
    echo $HUGEPAGES > /proc/sys/vm/nr_hugepages
    ALLOC_PAGES=$(cat /proc/sys/vm/nr_hugepages)
    echo "Allocated $ALLOC_PAGES / $HUGEPAGES"
    (( TRIES++ ))
  done

  if [ "$ALLOC_PAGES" -ne "$HUGEPAGES" ]; then
    echo "Hugepages allocation failed. Reverting..."
    echo 0 > /proc/sys/vm/nr_hugepages
    exit 1
  fi
}}

# -------------------------
# Event: PREPARE
# -------------------------
if [ "$command" = "prepare" ]; then

  # Allocate hugepages BEFORE isolating CPUs
  vfio-isolate \\
    drop-caches
  allocate_hugepages

  # Isolate VM cores but keep host on HOST_CORES
  vfio-isolate -u /tmp/vfio_isolate_history \\
    cpuset-modify --cpus C${{HOST_CORES}} /system.slice \\
    cpuset-modify --cpus C${{HOST_CORES}} /user.slice \\
    cpuset-modify --cpus C${{HOST_CORES}} /init.scope \\
    irq-affinity mask C${{VM_CORES}}

  echo "CPU isolation complete."
  echo "Host cores: $HOST_CORES"
  echo "VM cores:   $VM_CORES"

# -------------------------
# Event: STARTED
# -------------------------
elif [ "$command" = "started" ]; then
  # Set QEMU process to highest priority (nice -20)
  VM_NAME="$1"
  QEMU_PID=$(pgrep -f "qemu.*$VM_NAME")
  if [ -n "$QEMU_PID" ]; then
    renice -n -20 -p $QEMU_PID
    echo "Set QEMU process $QEMU_PID to nice -20"

    # Also set realtime scheduling for QEMU threads on VM cores
    for tid in $(ls /proc/$QEMU_PID/task/); do
      chrt -f -p 1 $tid 2>/dev/null || true
    done
    echo "Applied FIFO scheduling to QEMU threads"
  fi

# -------------------------
# Event: RELEASE
# -------------------------
elif [ "$command" = "release" ]; then
  echo "Releasing VFIO isolation..."
  vfio-isolate restore /tmp/vfio_isolate_history

  # Incase vfio-isolate restore failed
  vfio-isolate \\
    cpuset-modify --cpus C${{ALL_CORES}} /system.slice \\
    cpuset-modify --cpus C${{ALL_CORES}} /user.slice \\
    cpuset-modify --cpus C${{ALL_CORES}} /init.scope \\
    irq-affinity mask C${{ALL_CORES}}

  echo "Dropping hugepages..."
  echo 0 > /proc/sys/vm/nr_hugepages

  echo "Done."
fi
'''


def install_hook_script(
    allocation: CoreAllocation,
    vm_memory_mb: int,
    host_reserved_auto: bool = False
) -> bool:
    """Install the hook script to /etc/libvirt/hooks/qemu."""
    hook_path = Path("/etc/libvirt/hooks/qemu")
    hooks_dir = hook_path.parent

    # Check if running as root
    if os.geteuid() != 0:
        print("Error: Must run as root (sudo) to install hook script.", file=sys.stderr)
        print(f"Try: sudo {sys.argv[0]} --install-hook [options]", file=sys.stderr)
        return False

    # Create hooks directory if needed
    if not hooks_dir.exists():
        print(f"Creating directory: {hooks_dir}")
        hooks_dir.mkdir(parents=True, mode=0o755)

    # Generate and write script
    script_content = generate_hook_script(allocation, vm_memory_mb)

    # Backup existing hook if present
    if hook_path.exists():
        backup_path = hook_path.with_suffix(".bak")
        print(f"Backing up existing hook to: {backup_path}")
        hook_path.rename(backup_path)

    print(f"Writing hook script to: {hook_path}")
    hook_path.write_text(script_content)

    # Make executable
    hook_path.chmod(0o755)

    print("Hook script installed successfully!")
    print()
    print("Configuration:")
    print(f"  VM_CORES:   {allocation.vm_cores_str}")
    host_note = " (auto-detected)" if host_reserved_auto else ""
    print(f"  HOST_CORES: {allocation.host_cores_str}{host_note}")
    print(f"  ALL_CORES:  {allocation.all_cores_str}")
    print(f"  VM_MEMORY:  {vm_memory_mb} MB")
    print()
    print("You may need to restart libvirtd: sudo systemctl restart libvirtd")

    return True


def print_topology(topology: CPUTopology):
    """Print CPU topology information."""
    print("=" * 60)
    print("CPU TOPOLOGY")
    print("=" * 60)
    print(f"Sockets:          {topology.sockets}")
    print(f"Cores per socket: {topology.cores_per_socket}")
    print(f"Threads per core: {topology.threads_per_core}")
    print(f"Total threads:    {topology.total_threads}")

    if topology.is_hybrid:
        print()
        print("*** INTEL HYBRID CPU DETECTED ***")
        print(f"P-cores (Performance): {topology.p_cores_count}")
        print(f"E-cores (Efficiency):  {topology.e_cores_count}")
        print("Note: VM pinning will use P-cores only")
    print()

    print("Core -> Thread mapping:")
    print("-" * 40)
    for socket in range(topology.sockets):
        print(f"\nSocket {socket}:")
        # Show ALL cores with P/E labels on hybrid
        all_pairs = topology.get_all_core_pairs(socket)
        core_idx = 0
        for (sock, core_id), core in sorted(topology.cores.items()):
            if sock == socket:
                threads_str = ", ".join(str(t) for t in sorted(core.threads))
                if topology.is_hybrid:
                    core_type = "P" if core.is_p_core else "E"
                    print(f"  Core {core_idx:2d} [{core_type}]: threads [{threads_str}]")
                else:
                    print(f"  Core {core_idx:2d}: threads [{threads_str}]")
                core_idx += 1

    if topology.numa_nodes:
        print("\nNUMA Nodes:")
        print("-" * 40)
        for node, cpus in sorted(topology.numa_nodes.items()):
            print(f"  Node {node}: {format_cpu_list(cpus)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate CPU pinning configuration for VFIO/QEMU/libvirt VMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show CPU topology
  %(prog)s --show-topology

  # Generate config for 8-core VM with SMT (16 vCPUs), starting at core 2
  %(prog)s --vm-cores 8 --start-core 2

  # Generate config for 6-core VM without SMT
  %(prog)s --vm-cores 6 --vm-threads 1

  # Generate config for second socket/NUMA node
  %(prog)s --vm-cores 8 --socket 1

  # Install hook script to /etc/libvirt/hooks/qemu (requires sudo)
  sudo %(prog)s --install-hook --vm-cores 6 --vm-memory 16384

  # Output shell variables for bash script (use with eval)
  eval $(%(prog)s --output-vars --vm-cores 6)
        """
    )

    parser.add_argument("--show-topology", action="store_true",
                        help="Show CPU topology and exit")
    parser.add_argument("--output-vars", action="store_true",
                        help="Output VM_CORES, HOST_CORES, ALL_CORES as shell variables")
    parser.add_argument("--install-hook", action="store_true",
                        help="Install hook script to /etc/libvirt/hooks/qemu (requires sudo)")
    parser.add_argument("--vm-cores", type=int, default=4,
                        help="Number of physical cores for the VM. On hybrid CPUs, this is P-cores only (default: 4)")
    parser.add_argument("--vm-threads", type=int, default=2,
                        help="Threads per core (1=no SMT, 2=SMT) (default: 2)")
    parser.add_argument("--start-core", type=int, default=None,
                        help="First P-core to assign to VM (default: auto, after host-reserved)")
    parser.add_argument("--socket", type=int, default=0,
                        help="Socket/NUMA node to use (default: 0)")
    parser.add_argument("--host-reserved", type=int, default=None,
                        help="Number of P-cores reserved for host. On hybrid CPUs, E-cores are always assigned to host (default: auto)")
    parser.add_argument("--vm-memory", type=int, default=16384,
                        help="VM memory in MB for hugepages (default: 16384)")
    parser.add_argument("--iothreads", type=int, default=1,
                        help="Number of IOThreads (default: 1)")
    parser.add_argument("--no-emulator-pin", action="store_true",
                        help="Disable emulator pinning")

    args = parser.parse_args()

    topology = parse_cpu_topology()

    if args.show_topology:
        print_topology(topology)
        return

    # Generate core allocation (host_reserved and start_core auto-calculated if None)
    try:
        allocation = generate_core_allocation(
            topology,
            vm_cores=args.vm_cores,
            host_reserved_cores=args.host_reserved,
            socket=args.socket,
            start_core=args.start_core,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate effective host_reserved for display/XML generation
    effective_host_reserved = args.host_reserved
    if effective_host_reserved is None:
        effective_host_reserved = calculate_host_reserved_cores(topology, args.vm_cores, args.socket)

    # Install hook script mode
    if args.install_hook:
        if not install_hook_script(allocation, args.vm_memory, host_reserved_auto=(args.host_reserved is None)):
            sys.exit(1)
        return

    # Output shell variables mode (for bash scripts)
    if args.output_vars:
        output_shell_vars(allocation)
        return

    # Full output mode
    print_topology(topology)

    print("=" * 60)
    print("CORE ALLOCATION")
    print("=" * 60)
    print(f'VM_CORES="{allocation.vm_cores_str}"')
    print(f'HOST_CORES="{allocation.host_cores_str}"')
    print(f'ALL_CORES="{allocation.all_cores_str}"')
    print()

    # Calculate effective start_core
    effective_start_core = args.start_core if args.start_core is not None else effective_host_reserved

    # Validate inputs for XML (use P-cores count on hybrid)
    max_cores = topology.p_cores_count if topology.is_hybrid else topology.cores_per_socket
    if effective_start_core + args.vm_cores > max_cores:
        core_type = "P-cores" if topology.is_hybrid else "cores"
        print(f"Error: Requested cores exceed available {core_type}. "
              f"Max start_core for {args.vm_cores} VM cores: {max_cores - args.vm_cores}",
              file=sys.stderr)
        sys.exit(1)

    if args.vm_threads > topology.threads_per_core:
        print(f"Warning: VM threads ({args.vm_threads}) exceeds host threads per core "
              f"({topology.threads_per_core}). Adjusting to {topology.threads_per_core}.",
              file=sys.stderr)
        args.vm_threads = topology.threads_per_core

    # Generate configuration
    print("=" * 60)
    print("LIBVIRT XML CONFIGURATION")
    print("=" * 60)
    core_label = "P-cores" if topology.is_hybrid else "cores"
    print(f"VM Configuration: {args.vm_cores} {core_label} x {args.vm_threads} threads = {args.vm_cores * args.vm_threads} vCPUs")
    print(f"Using {core_label} {effective_start_core}-{effective_start_core + args.vm_cores - 1} on socket {args.socket}")
    if args.host_reserved is None:
        print(f"Host reserved: {effective_host_reserved} {core_label} (auto-detected)")
    else:
        print(f"Host reserved: {effective_host_reserved} {core_label}")
    if topology.is_hybrid:
        print(f"E-cores: All {topology.e_cores_count} assigned to host")
    print()
    print("Add the following to your libvirt VM XML:")
    print("-" * 60)

    xml = generate_libvirt_xml(
        topology,
        vm_cores=args.vm_cores,
        vm_threads=args.vm_threads,
        start_core=effective_start_core,
        socket=args.socket,
        emulator_pin=not args.no_emulator_pin,
        iothreads=args.iothreads,
        host_reserved_cores=effective_host_reserved,
    )
    print(xml)



if __name__ == "__main__":
    main()
