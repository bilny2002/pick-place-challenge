"""Franka + Robotiq "place the ball in the bowl" sandbox on mjlab.

Two modules: ``scene`` (all physics/scene construction + asset fetching) and
``task`` (the MDP, the env config, and task registration). Import
``pick_place_challenge.task`` to register the ``Mjlab-PlaceBall-Franka-*`` tasks.
"""
