import numpy as np

class Worlds:
    categories = [i for i in range(9)]
    colors = ["white", "blue", "orange", "yellow", "green"]
    #  0      1     2     3     4
    # up, right, down, left, stay
    act_map = [[-1,0],[0,1],[1,0],[0,-1],[0,0]]
    act_name = ["UP","RIGHT","DOWN","LEFT","STAY"]
   
    max_idx = 6

    @staticmethod
    def define_worlds():
        """
        Map state categories to states
        First, create just a map of the indexes
        Categories are integers on the interval [0,N]
        want matrix m such that r*m = reward function
        """
        grid_maps = list()
        state_starts = list()
        viz_starts = list()
        grid_maps.append(np.array([
            [0,0,1,1,0,0,0,2,2,4],
            [0,0,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,3,3,3,3,3,3,0,0],
            [0,0,0,0,3,3,3,3,0,0]]))
        state_starts.append([4,3])
        viz_starts.append([4,3])

        grid_maps.append(np.array([
            [4,3,3,3,3,0,2,2,0,0],
            [0,3,3,3,3,0,2,2,0,0],
            [0,0,1,1,0,0,2,2,0,0],
            [0,0,1,1,0,0,2,2,0,0],
            [0,0,0,0,0,0,0,0,0,0]]))
        state_starts.append([4,0])
        viz_starts.append([4,0])

        grid_maps.append(np.array([
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,2,2,0,3,3,0,0,0],
            [0,0,2,2,4,3,3,0,0,0],
            [0,0,2,2,1,1,1,0,0,0],
            [0,0,0,0,1,1,1,0,0,0]]))
        state_starts.append([4,0])
        viz_starts.append([4,0])

        grid_maps.append(np.array([
            [2,4],
            [0,1]]))
        state_starts.append([1,0])
        viz_starts.append([1,0])

        ## Idx 4
        grid_maps.append(np.array([
            [1,2,4],
            [2,3,1],
            [0,1,3]]))
        state_starts.append([1,0])
        viz_starts.append([1,0])
        
        ## Idx 5
        grid_maps.append(np.array([
            [7,8,4],
            [3,5,6],
            [0,1,2]]))
        state_starts.append([1,0])
        viz_starts.append([1,0])
        
        ## Idx 6 (pairs with 4)
        grid_maps.append(np.array([
            [1,1,2,2,4,4],
            [1,1,2,2,4,4],
            [2,2,3,3,1,1],
            [2,2,3,3,1,1],
            [0,0,1,1,3,3],
            [0,0,1,1,3,3]]))
        state_starts.append([1,0])
        viz_starts.append([1,0])
        return grid_maps, state_starts, viz_starts

    @classmethod
    def get_worlds(cls, worlds, starts, viz_starts):
        return {
            'grid_maps': worlds,
            'init_states': starts,
            'viz_init_states': viz_starts
        }
