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
static PyObject *djik(PyObject *self, PyObject *args) {
  const PyArrayObject* lengths_object;
  const PyArrayObject* locations_object;
  const PyArrayObject* neighbors_object;
  int m;
  int n;
  int goal;

  if (!PyArg_ParseTuple(
        args, "OOOiii", // i = int, O = object
        &lengths_object,
        &locations_object,
        &neighbors_object,
        &m, &n, &goal
        ))
    return NULL;

  
  int* lengths = (int*) lengths_object->data;
  int* locations = (int*) locations_object->data;
  int* neighbors = (int*) neighbors_object->data;

  

  float* costs = new float[n];

  for (int i = 0; i < n; ++i)
  {
    costs[i] = INF;
  }




    std::priority_queue<Node> nodes_to_visit;
    costs[goal] = 0;
    nodes_to_visit.push(Node(goal, 0, 1, 0));
      
    while (!nodes_to_visit.empty()) {
        Node cur = nodes_to_visit.top();
        nodes_to_visit.pop();

        for (int i = 0; i < lengths[cur.idx]; i++) {
        
            int neighbor = neighbors[locations[cur.idx] + i];
            float new_cost = 1 + costs[cur.idx];
        
        
        
            if (new_cost < costs[neighbor]) {
                // estimate the cost to the goal based on legal moves
                // Get the heuristic method to use

                // paths with lower expected cost are explored first
                float priority = new_cost;
                costs[neighbor] = new_cost;

                nodes_to_visit.push(Node(neighbor, priority, cur.path_length+1, 0));

            }
        }
    }

  
  npy_intp dims[1] = {n};
  PyArrayObject* cluster_assignment = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, costs);
  
  
  return PyArray_Return(cluster_assignment);
}






static PyMethodDef djik_methods[] = {
    {"djik", (PyCFunction)djik, METH_VARARGS, "djik"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef djik_module = {
    PyModuleDef_HEAD_INIT,"djik", NULL, -1, djik_methods
};

PyMODINIT_FUNC PyInit_djik(void) {
  import_array();
  return PyModule_Create(&djik_module);
}

