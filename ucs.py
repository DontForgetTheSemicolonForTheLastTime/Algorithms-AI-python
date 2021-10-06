nimport Queue as Q

# max_size = 5
# q=Q.PriorityQueue(max_size)

# # q.put((priority_value,data))

# q.put((3,3))
# q.put((4,3))
# q.put((1,"tutorial Example"))

# print(q.get())

# -*- coding: utf-8 -*-

class Node:
    def __init__(self, state=None, parent=None):
        self.state = state
        self.parent = parent
        self.children = []

    def addChildren(self, children):
        self.children.extend(children)

# this function is to basically find the children of a node.
# if either m or n is the parent node (node), then append the other one into the children list
def expandAndReturnChildren(state_space, node):
    children = []
    # for each element in the list of (initial_state (m), goal_state (n), cost (c)
    for [m, n, c] in state_space:
        if m == node.state:
            children.append(Node(n, node.state))
        elif n == node.state:
            children.append(Node(m, node.state))
    return children

# this function is to carry out the breadth first search
# arguments are the list of state space, the initial state and the goal state
def ucs(state_space,
    explored = []
    frontier = []
    q = Q.PriorityQueue()
    q.put((0, v, [v]))
    goalie = Node()
    found_goal = False
    while not q.empty():
        frontier, explored, path = q.get()
        if explored not in frontier:
            frontier.append(Node(initial_state, None))
                if frontier == goal_state:
                    found_goal = True
                    goalie = child
                else:
                    for frontier in explored :
                        
    
    solution = []
    # the first value in frontier list is the initial_state
    frontier.append(Node(initial_state, None))

    # while goal_state: "Bucharest" is not reached, loop the following:
    while not found_goal:
        # run the function to expand the first node and store its child node in children[]
        children = expandAndReturnChildren(state_space, frontier[0])
        frontier[0].addChildren(children)
        # append the first node to explored[]
        explored.append(frontier[0])
        # delete the first node from frontier[]
        del frontier[0]

        # for loop to see check if child node is the goal_state or not
        for child in children:
            if not (child.state in [e.state for e in explored]) and not (child.state in [f.state for f in frontier]):
                # if child node is the goal_state, then turn found_goal boolean to true and append child node
                # to frontier[]
                # the goal node will be stored in goalie
                if child.state == goal_state:
                    found_goal = True
                    goalie = child
                frontier.append(child)

        print("Explored:", [e.state for e in explored])
        print("Frontier:", [f.state for f in frontier])
        print("Children:", [c.state for c in children])
        print("")

    solution = [goalie.state]
    path_cost = 0
    # while loop will continue to loop until the initial_state has been traced back, which is also why condition
    # is set as goalie.parent is not None. because the initial_state object has no parent as it is the first node
    # .parent is a property to let us trace back to find an object's parent (predecessor)
    while goalie.parent is not None:
        # insert the goalie's parent object into solution[]
        solution.insert(0, goalie.parent)
        # for each element in explored[]
        for e in explored:
            # if object at e is goalie's parent, then replace goalie with the parent node and repeat the process
            # this process allow us to path cost of parent node, and its parent node, and so on, until the initial node
            if e.state == goalie.parent:
                path_cost += getCost(state_space, e.state, goalie.state)
                goalie = e
                break

    return solution, path_cost


def getCost(state_space, state0, state1):
    for [m, n, c] in state_space:
        if [state0, state1] == [m, n] or [state1, state0] == [m, n]:
            return c


if __name__ == '__main__':
    state_space = [['Arad', 'Zerind', 75], ['Arad', 'Timisoara', 118], ['Arad', 'Sibiu', 140],
                   ['Timisoara', 'Lugoj', 111], ['Lugoj', 'Mehadia', 70], ['Mehadia', 'Drobeta', 75],
                   ['Drobeta', 'Craiova', 120], ['Craiova', 'Rimnicu Vilcea', 146], ['Craiova', 'Pitesti', 138],
                   ['Pitesti', 'Bucharest', 101], ['Rimnicu Vilcea', 'Pitesti', 97], ['Rimnicu Vilea', 'Sibiu', 80],
                   ['Zerind', 'Arad', 75], ['Zerind', 'Oradea', 71], ['Oradea', 'Sibiu', 151], ['Sibiu', 'Arad', 140],
                   ['Sibiu', 'Fagaras', 99], ['Sibiu', 'Rimnicu Vilcea', 80], ['Fagaras', 'Bucharest', 211]]

    initial_state = 'Arad'
    goal_state = 'Bucharest'

    [solution, cost] = bfs(state_space, initial_state, goal_state)
    print("Solution:", solution)
    print("Path cost:", cost)
