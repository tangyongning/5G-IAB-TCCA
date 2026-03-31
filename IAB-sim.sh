# wscript configuration for 5G-LENA with IAB
def build(bld):
    module = bld.create_module(
        name = 'nr-iab-fault-localization',
        source = [
            'model/iab-fault-localization.cc',
            'helper/iab-topology-helper.cc',
        ],
        headers = [
            'model/iab-fault-localization.h',
        ],
        use = ['nr', 'point-to-point', 'applications', 'flow-monitor'],
    )
