import networkx as nx
import gurobipy 
import pickle
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('in_dir', None, 'Input directory containing gpickles files')
flags.DEFINE_string('out_dir', None, 'Output directory for .lp and .pkl files')


def constr_name(v1, v2):
    if v1 < v2:
        name = 'c_%d_%d' %(v1, v2)
    else:
        name = 'c_%d_%d' %(v2, v1)

    return name

    
def convert_mvc(graph):
    """Convert a networkx graph into an MVC integer program.

    Arguments:
      graph: a networkx graph.
    
    Returns: 
      var_dict: a dict mapping node index to variable name.
      model: an integer program representing the MVC problem.
    """
    m = gurobipy.Model()
    var_dict = {}
    obj = 0
    
    for v1, v2 in graph.edges():
        if not v1 in var_dict:
            # var_v1 = m.addVar(lb=0, ub=1, vtype=gurobipy.GRB.INTEGER, name='v_%d' %v1)
            var_v1 = m.addVar(vtype=gurobipy.GRB.BINARY, name='v_%d' %v1)            
            var_dict[v1] = 'v_%d' %v1
            obj += var_v1
        else:
            var_v1 = m.getVarByName(var_dict[v1])

        if not v2 in var_dict:
            # var_v2 = m.addVar(lb=0, ub=1, vtype=gurobipy.GRB.INTEGER, name='v_%d' %v2)
            var_v2 = m.addVar(vtype=gurobipy.GRB.BINARY, name='v_%d' %v2)            
            var_dict[v2] = 'v_%d' %v2
            obj += var_v2
        else:
            var_v2 = m.getVarByName(var_dict[v2])            

        m.addConstr(var_v1 + var_v2 >= 1, constr_name(v1, v2))
        m.update()

    m.setObjective(obj, gurobipy.GRB.MINIMIZE)

    return var_dict, m


def write_mvc_lp(in_file, out_file_prefix):
    """Write the constructed model from an in_file to an out_file.

    Arguments:
      in_file: the path to a gpickle file.
      out_file: the path to an output file.
    """

    graph = nx.read_gpickle(in_file)
    var_dict, m = convert_mvc(graph)
    m.write(out_file_prefix + ".lp")
    out_file = open(out_file_prefix + ".pkl", "wb")
    pickle.dump(var_dict, out_file)
    out_file.close()


def main(argv):
    in_dir = FLAGS.in_dir
    out_dir = FLAGS.out_dir

    for gpickle_f in os.listdir(in_dir):
        abs_filename = os.path.join(in_dir, gpickle_f)
        write_mvc_lp(abs_filename, os.path.join(out_dir, gpickle_f.split('.')[0]))
    
if __name__ == "__main__":
    app.run(main)
