from retroRL.prioritizers.prioritizer import Prioritizer

class DefaultPrioritizer(Prioritizer):
    '''
    Will not prioritize the objects in any way. Returns a priority of 1 for any object. Can therefore be used both for (non)-
    prioritization of templates and/or retro-synthetic precursors.
    '''

    def __init__(self):
        pass
    def get_priority(self, object_to_prioritize, **kwargs):
        try:
            (templates, target) = object_to_prioritize
            return templates
        # if not a tuple: prioritization of a retro-precursor element.
        except TypeError:
            return 1.0

    def load_model(self):
        pass
