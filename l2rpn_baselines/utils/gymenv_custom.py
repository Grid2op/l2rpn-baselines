# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from abc import abstractmethod
from typing import Tuple, Dict, List
import numpy as np

from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from grid2op.gym_compat import GymEnv
from grid2op.gym_compat.utils import GYM_VERSION

from packaging import version


class GymEnvWithHeuristics(GymEnv):
    """This abstract class is used to perform some actions, independantly of a RL
    agent on a grid2op environment.
    
    It can be used, for example, to train an agent (for example a deep-rl agent)
    if you want to use some heuristics at inference time (for example
    you reconnect every powerline that you can.)
    
    The heuristic you want to implement should be implemented in :func:`GymEnvWithHeuristics.heuristic_actions`.
    
    Examples
    --------
    Let's imagine, for example, that you want to implement an RL agent that performs actions on the grid. But you noticed
    that your agent performs better if the all the powerlines are reconnected (which is often the case by the way).
    
    To that end, you want to force the reconnection of powerline each time it's possible. When it's not possible, you want to
    let the neural network do what is best for the environment.
    
    Training an agent on such setting might be difficult and require recoding some (deep) part of the training framework (*eg*
    stable-baselines). Unless... You use a dedicated "environment".
    
    In this environment (compatible, inheriting the base class `gym.Env`) will handle all the "heuristic" part and only show
    the agent with the state where it should act.
    
    Basically a "step" happens like this:
    
    #. the agent issue an action (gym format)
    #. the action (gym format) is decoded to a grid2op compatible action (thanks to the action_space)
    #. this grid2op action is implemented on the grid (thanks to the underlying grid2op environment)
       and the corresponding grid2op observation is generated
    #. this observation is processed by the  :func:`GymEnvWithHeuristics.apply_heuristics_actions`: the grid2op_env.step
       is called until the NN agent is require to take a decision (or the flag `done=True` is set).
    #. the observation (corresponding to the last step above) is then converted to a gym action (thanks to the observation_space)
       which is forwarded to the agent.
    
    The agent then only "sees" what is not processed by the heuristic. It is trained only on the relevant "state".

    """
    POSSIBLE_REWARD_CUMUL = ["init", "last", "sum", "max"]
    def __init__(self, env_init, *args, reward_cumul="last", **kwargs):
        super().__init__(env_init, *args, **kwargs)
        self._reward_cumul = reward_cumul
        
        if not self._reward_cumul in type(self).POSSIBLE_REWARD_CUMUL:
            raise RuntimeError("Wrong argument for the reward_cumul parameters. "
                               f"You provided \"{self._reward_cumul}\" (possible "
                               f"values are {type(self).POSSIBLE_REWARD_CUMUL}).")
        self.min_gym_version = version.parse("0.26.0")
            
    @abstractmethod
    def heuristic_actions(self,
                          g2op_obs: BaseObservation,
                          reward: float,
                          done: bool,
                          info: Dict) -> List[BaseAction]:
        """This function has the same signature as the "agent.act"  function. It allows to implement a heuristic.
        
        It can be called multiple times per "gymenv step" and is expect to return a list of grid2op actions (in the
        correct order) to be done on the underlying grid2op environment. 

        An implementation of such a function (for example) can be found at :func:`GymEnvWithReco.heuristic_actions` or
        :func:`GymEnvWithRecoWithDN.heuristic_actions`
        
        This function can return a list of action that will "in turn" be executed on the grid. It is only after each 
        and every actions that are returned that this function is called again.
        
        .. note::
            You MUST return "[do_nothing]" if your heuristic chose to do nothing at a certain step. Otherwise (if
            the returned list is empty "[]" the agent is asked to perform an action.)
        
        .. note::
            We remind that inside a "gym env" step, a lot of "grid2op env" steps might be happening.
            
            As long as a heuristic action is selected (ie as long as this function does not return the empty list)
            this action is performed on the grid2op environment.
            
        Parameters
        ----------
        g2op_obs : BaseObservation
            [description]
        reward : float
            The last reward the agent (or the heuristic) had.
            This is the `reward` part of the last call to `obs, reward, done, info = grid2op_env.step(grid2op_act)`
        done : bool
            Whether the environment is "done" or not. It should be "False" in most cases. 
            This is the `done` part of the last call to `obs, reward, done, info = grid2op_env.step(grid2op_act)`
        info : Dict
            `info` part of the last call to `obs, reward, done, info = grid2op_env.step(grid2op_act)`

        Returns
        -------
        List[BaseAction]
            The ordered list of actions to implement, selected by the "heuristic" / "expert knowledge" / "automatic action".
        """
        return []
    
    def apply_heuristics_actions(self,
                                 g2op_obs: BaseObservation,
                                 reward: float,
                                 done: bool,
                                 info: Dict ) -> Tuple[BaseObservation, float, bool, Dict]:
        """This function implements the "logic" behind the heuristic part. Unless you have a particular reason too, you
        probably should not modify this function.
        
        If you modify it, you should also modify the way the agent implements it (remember: this function is used 
        at training time, the "GymAgent" part is used at inference time. Both behaviour should match for the best
        performance).

        While there are "heuristics" / "expert rules" / etc. this function should perform steps in the underlying grid2op
        environment.
        
        It is expected to return when:
        
        - either the flag `done` is ``True`` 
        - or the neural network agent is asked to perform action on the grid
        
        The neural network agent will receive the outpout of this function. 
        
        Parameters
        ----------
        g2op_obs : BaseObservation
            The grid2op observation.
            
        reward : ``float``
            The reward
            
        done : ``bool``
            The flag that indicates whether the environment is over or not.
            
        info : Dict
            Other information flags

        Returns
        -------
        Tuple[BaseObservation, float, bool, Dict]
            It should return `obs, reward, done, info`(same as a single call to `grid2op_env.step(grid2op_act)`)
            
            Then, this will be transmitted to the neural network agent (but before the observation will be 
            transformed to a gym observation thanks to the observation space.)
            
        """
        need_action = True
        res_reward = reward
        
        tmp_reward = reward
        tmp_info = info
        while need_action:
            need_action = False
            g2op_actions = self.heuristic_actions(g2op_obs, tmp_reward, done, tmp_info)
            for g2op_act in g2op_actions:
                need_action = True
                tmp_obs, tmp_reward, tmp_done, tmp_info = self.init_env.step(g2op_act)
                g2op_obs = tmp_obs
                done = tmp_done
                
                if self._reward_cumul == "max":
                    res_reward = max(tmp_reward, res_reward)
                elif self._reward_cumul == "sum":
                    res_reward += tmp_reward
                elif self._reward_cumul == "last":
                    res_reward = tmp_reward
                    
                if tmp_done:
                    break
            if done:
                break
        return g2op_obs, res_reward, done, info
    
    def fix_action(self, grid2op_action, g2op_obs):
        """This function can be used to "fix" / "modify" / "cut" / "change"
        a grid2op action just before it will be applied to the underlying "env.step(...)"
        
        This can be used, for example to "limit the curtailment or storage" of the
        action in case this one is too strong and would lead to a game over.

        By default it does nothing.
        
        Parameters
        ----------
        grid2op_action : _type_
            _description_
            
        """
        return grid2op_action
    
    def step(self, gym_action):
        """This function implements the special case of the "step" function (as seen by the "gym environment") that might
        call multiple times the "step" function of the underlying "grid2op environment" depending on the
        heuristic.
        
        It takes a gym action, convert it to a grid2op action (thanks to the action space).
        
        Then process the heuristics / expert rules / forced actions / etc. and return the next gym observation that will
        be processed by the agent.
        
        The number of "grid2op steps" can vary between different "gym environment" call to "step".
        
        It has the same signature as the `gym.Env` "step" function, of course. 

        Parameters
        ----------
        gym_action :
            the action (represented as a gym one) that the agent wants to perform.

        Returns
        -------
        gym_obs:
            The gym observation that will be processed by the agent
            
        reward: ``float``
            The reward of the agent (that might be computed by the )
            
        done: ``bool``
            Whether the episode is over or not
            
        info: Dict
            Other type of informations
            
        """
        g2op_act_tmp = self.action_space.from_gym(gym_action)
        g2op_act = self.fix_action(g2op_act_tmp, self._previous_act)
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        if not done:
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
        self._previous_act = g2op_obs
        gym_obs = self.observation_space.to_gym(g2op_obs)
        if hasattr(type(self), "_gymnasium") and type(self)._gymnasium:
            truncated = False
            return gym_obs, float(reward), done, truncated, info
        else:
            return gym_obs, float(reward), done, info
        
    def reset(self, *, seed=None, return_info=False, options=None):
        """This function implements the "reset" function. It is called at the end of every episode and
        marks the beginning of a new one.
        
        Again, before the agents sees any observations from the environment, they are processed by the 
        "heuristics" / "expert rules".
        
        .. note::
            The first observation seen by the agent is not necessarily the first observation of the grid2op environment.

        Returns
        -------
        gym_obs:
            The first open ai gym observation received by the agent
        """
        is_gymnasium = hasattr(type(self), "_gymnasium") and type(self)._gymnasium
        if is_gymnasium:
            return_info = True
            
        done = True
        info = {}  # no extra information provided !
        while done:
            if is_gymnasium or GYM_VERSION >= self.min_gym_version:
                super().reset(seed=seed, options=options) # reset the scenario
            else:
                super()._aux_reset(seed, return_info, options)  # reset the scenario
            g2op_obs = self.init_env.get_obs()  # retrieve the observation
            reward = self.init_env.reward_range[0]  # the reward at first step is always minimal
            
            # perform the "heuristics" steps
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, False, info)
            
            # convert back the observation to gym
            if not done:
                self._previous_act = g2op_obs
                gym_obs = self.observation_space.to_gym(g2op_obs)
                break
            
        if return_info:
            return gym_obs, info
        else:
            return gym_obs
    
class GymEnvWithReco(GymEnvWithHeuristics):
    """This specific type of environment with "heuristics" / "expert rules" / "expert actions" is an
    example to illustrate how to perfom an automatic powerline reconnection.
    
    For this type of environment the only heuristic implemented is the following: "each time i can
    reconnect a powerline, i don't ask the agent, i reconnect it and send it the state after the powerline
    has been reconnected".

    With the proposed class, implementing it is fairly easy as shown in function :func:`GymEnvWithReco.heuristic_actions`
    
    """
    def heuristic_actions(self, g2op_obs, reward, done, info) -> List[BaseAction]:
        """The heuristic is pretty simple: each there is a powerline with a cooldown at 0 and that is disconnected
        the heuristic reconnects it.

        Parameters
        ----------
        See parameters of :func:`GymEnvWithHeuristics.heuristic_actions`

        Returns
        -------
        See return values of :func:`GymEnvWithHeuristics.heuristic_actions`
        """
        
        # computes which powerline can be reconnected
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        res = []
        if np.any(to_reco):
            # If I can reconnect any, I do it
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space({"set_line_status": [(line_id, +1)]})
                res.append(g2op_act)
        return res
        
    
class GymEnvWithRecoWithDN(GymEnvWithHeuristics):
    """This environment is slightly more complex that the other one.
    
    It consists in 2 things:
    
    #. reconnecting the powerlines if possible
    #. doing nothing is the state of the grid is "safe" (for this class, the notion of "safety" is pretty simple: if all
       flows are bellow 90% (by default) of the thermal limit, then it is safe)
    
    If for a given step, non of these things is applicable, the underlying trained agent is asked to perform an action
    
    .. warning::
        When using this environment, we highly recommend to adapt the parameter `safe_max_rho` to suit your need.
        
        Sometimes, 90% of the thermal limit is too high, sometimes it is too low.
        
    """
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, **kwargs):
        super().__init__(env_init, reward_cumul=reward_cumul, *args, **kwargs)
        self._safe_max_rho = safe_max_rho
        
    def heuristic_actions(self, g2op_obs, reward, done, info) -> List[BaseAction]:
        """To match the description of the environment, this heuristic will:
        
        - return the list of all the powerlines that can be reconnected if any
        - return the list "[do nothing]" is the grid is safe
        - return the empty list (signaling the agent should take control over the heuristics) otherwise

        Parameters
        ----------
        See parameters of :func:`GymEnvWithHeuristics.heuristic_actions`

        Returns
        -------
        See return values of :func:`GymEnvWithHeuristics.heuristic_actions`
        """
        
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        res = []
        if np.any(to_reco):
            # reconnect something if it can be
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space({"set_line_status": [(line_id, +1)]})
                res.append(g2op_act)
        elif g2op_obs.rho.max() <= self._safe_max_rho:
            # play do nothing if there is "no problem" according to the "rule of thumb"
            res = [self.init_env.action_space()]
        return res
