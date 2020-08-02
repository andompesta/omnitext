
CONLL_LABLES = dict(ORG=0,
                    PER=1,
                    LOC=2,
                    MISC=3)
CONLL_IDX_LABEL = dict([(k, v) for v, k in CONLL_LABLES.items()])
CONLL_LABEL_QUERY = dict(ORG="organization entities are limited to named corporate, governmental, or other "
                             "organizational entities.",
                         PER="person entities are named persons or family.",
                         LOC="location entities are the name of politically or geographically defined locations such "
                             "as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
                         MISC="examples of miscellaneous entities include events, nationalities, products and works of "
                              "art.")

ACE05_LABELS = dict(ORG=0,
                    PER=1,
                    LOC=2,
                    GPE=3,
                    FAC=4,
                    VEH=5,
                    WEA=6)
ACE05_IDX_LABELS = dict([(k, v) for v, k in ACE05_LABELS.items()])
ACE05_LABELS_QUERY = dict(
    FAC="facility entities are limited to buildings and other permanent man-made structures such as buildings, "
        "airports, highways, bridges.",
    GPE="geographical political entities are geographical regions defined by political and or social groups such as"
        " countries, nations, regions, cities, states, government and its people.",
    LOC="location entities are limited to geographical entities such as geographical areas and landmasses, mountains,"
        " bodies of water, and geological formations.",
    ORG="organization entities are limited to companies, corporations, agencies, institutions and other groups of "
        "people.",
    PER="a person entity is limited to human including a single individual or a group.",
    VEH="vehicle entities are physical devices primarily designed to move, carry, pull or push the transported object "
        "such as helicopters, trains, ship and motorcycles.",
    WEA="weapon entities are limited to physical devices such as instruments for physically harming such as guns, arms"
        " and gunpowder."
)

__all__ = [
    "CONLL_LABLES",
    "CONLL_IDX_LABEL",
    "CONLL_LABEL_QUERY",
    "ACE05_LABELS",
    "ACE05_IDX_LABELS",
    "ACE05_LABELS_QUERY"
]