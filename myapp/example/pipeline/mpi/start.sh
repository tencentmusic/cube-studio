#!/bin/bash
echo "OMPI_MCA_plm_rsh_agent" $OMPI_MCA_plm_rsh_agent
echo "OMPI_MCA_orte_default_hostfile" $OMPI_MCA_orte_default_hostfile

cp /examples/tensorflow2_mnist.py ./
# mpirun -np "2" --allow-run-as-root -bind-to none -map-by slot -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python /examples/tensorflow2_mnist.py
