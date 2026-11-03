"""Fixed bilingual category normalization, defined before evaluation."""
import re

ALIASES={
 'chair':['椅','凳','chair','stool'], 'cup':['杯','cup','mug'],
 'bicycle':['自行车','单车','脚踏车','bicycle','bike'],
 'skateboard':['滑板','skateboard'], 'oven':['烤箱','烤炉','oven'],
 'couch':['沙发','couch','couches','sofa'], 'bowl':['碗','bowl'],
 'motorcycle':['摩托','motorcycle','motorbike'], 'snowboard':['滑雪板','snowboard'],
 'microwave':['微波炉','microwave'], 'cat':['猫','cat'],
 'donut':['甜甜圈','甜圈','多纳圈','donut','doughnut'], 'keyboard':['键盘','keyboard'],
 'orange':['橙子','橘子','桔子','柑橘','orange','mandarin'], 'pizza':['披萨','比萨','pizza'],
}


def matched_categories(names):
    def matches(alias,name):
        if alias=='orange':return re.fullmatch(r'(?:an? |the )?oranges?(?: fruit)?',name.strip().lower()) is not None
        if alias=='猫' and any(word in name for word in ('猫头鹰','熊猫','猫鼬')):return False
        if alias.isascii():return re.search(r'(?<![a-z])'+re.escape(alias)+r's?(?![a-z])',name.lower()) is not None
        return alias in name
    return sorted(c for c,aliases in ALIASES.items() if any(any(matches(a,name) for a in aliases) for name in names))
