#include <queue>
#include <limits>
#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <experimental_heuristics.h>
#include<map>
#include <string>


const float INF = std::numeric_limits<float>::infinity();

// represents a single pixel
class Node {
  public:
    int pidx; // previous index
    int idx; // index in the flattened grid
    float cost; // cost of traversing this pixel
    int path_length; // the length of the path to reach this node
    int cluster;
    Node() : idx(0), cost(INF), path_length(1), cluster(0) {}
    Node(int i, float c, int path_length, int cc) : idx(i), cost(c), path_length(path_length), cluster(cc) {}
};

// the top of the priority queue is the greatest element by default,
// but we want the smallest, so flip the sign
bool operator<(const Node &n1, const Node &n2) {
  return n1.cost > n2.cost;
}

// See for various grid heuristics:
// http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#S7
// L_\inf norm (diagonal distance)
inline float linf_norm(int i0, int j0, int i1, int j1) {
  return std::max(std::abs(i0 - i1), std::abs(j0 - j1));
}

// L_1 norm (manhattan distance)
inline float l1_norm(int i0, int j0, int i1, int j1) {
  return std::abs(i0 - i1) + std::abs(j0 - j1);
}




// weights:        flattened h x w grid of costs
// h, w:           height and width of grid
// start, goal:    index of start/goal in flattened grid
// diag_ok:        if true, allows diagonal moves (8-conn.)
// paths (output): for each node, stores previous node in path
static PyObject *dclust(PyObject *self, PyObject *args) {
  const PyArrayObject* edges_object;
  const PyArrayObject* values_object;
  const PyArrayObject* lengths_object;
  const PyArrayObject* locations_object;
  const PyArrayObject* neighbors_object;
  const PyArrayObject* active_object;
  int m;
  int n;
  int num_clusters;
  int num_actives;

  if (!PyArg_ParseTuple(
        args, "OOOOOOiiii", // i = int, O = object
        &edges_object,
        &values_object,
        &lengths_object,
        &locations_object,
        &neighbors_object,
        &active_object,
        &m, &n, &num_clusters, &num_actives
        ))
    return NULL;

  int* edges = (int*) edges_object->data;
  float* values = (float*) values_object->data;
  int* lengths = (int*) lengths_object->data;
  int* locations = (int*) locations_object->data;
  int* neighbors = (int*) neighbors_object->data;
  int* active_nodes = (int*) active_object->data;

  //PyErr_SetString(PyExc_Warning, (std::to_string(starts[0]) + std::to_string(starts[1]) + std::to_string(starts[2]) + std::to_string(starts[3]) + std::to_string(starts[4])).c_str());
  //PyErr_Print();
  std::map<std::pair<int, int>, float> value_map;

  for (int i=0; i<m; i++)
  {
      value_map.insert({{edges[i*2], edges[i*2+1]}, values[i]});
  }
  //return NULL;
  int* paths = new int[n];
  int path_length = -1;

  int* clust_assign = new int[n];
  float* costs = new float[n];
  //Node* nodes = new Node[n];

  for (int i = 0; i < n; ++i)
  {
    costs[i] = 1000000;
    clust_assign[i] = 0;
    //Node current_node(i, INF, 1, 0);
    //nodes[i] = current_node;
  }


  for (int k = 1; k<num_clusters+1; k++){
      int arg = active_nodes[0];
      float max = -1;

      
      for (int a = 0; a<num_actives; a++){
          int node_id = active_nodes[a];
          if (costs[node_id] >= max){
              arg = node_id;
              max = costs[node_id];
          }
      }
      

      std::priority_queue<Node> nodes_to_visit;
      clust_assign[arg] = k;
      costs[arg] = 0;
      nodes_to_visit.push(Node(arg, 0, 1, k));
      
      while (!nodes_to_visit.empty()) {
          Node cur = nodes_to_visit.top();
          nodes_to_visit.pop();

          for (int i = 0; i < lengths[cur.idx]; i++) {
        
            int neighbor = neighbors[locations[cur.idx] + i];
            float new_cost = 0.1 + value_map[{cur.idx, neighbor}] + costs[cur.idx];
            float nc = 0;
        
        
        
            if (new_cost < costs[neighbor]) {
              // estimate the cost to the goal based on legal moves
              // Get the heuristic method to use

              // paths with lower expected cost are explored first
              float priority = new_cost;
              costs[neighbor] = new_cost;
              clust_assign[neighbor] = k;

              nodes_to_visit.push(Node(neighbor, priority, cur.path_length+1, k));

            }
          
        }

      }


  }
  

  

  
  npy_intp dims[1] = {n};
  PyArrayObject* cluster_assignment = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_INT32, clust_assign);
  
  
  return PyArray_Return(cluster_assignment);
}






static PyMethodDef dclust_methods[] = {
    {"dclust", (PyCFunction)dclust, METH_VARARGS, "dclust"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef dclust_module = {
    PyModuleDef_HEAD_INIT,"dclust", NULL, -1, dclust_methods
};

PyMODINIT_FUNC PyInit_dclust(void) {
  import_array();
  return PyModule_Create(&dclust_module);
}

