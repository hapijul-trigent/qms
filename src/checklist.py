from collections import defaultdict

CHECKLIST = dict()
TOP_CHECKLIST = defaultdict(list)
SIDE_CHECKLIST = defaultdict(list)
BOTTOM_CHECKLIST = defaultdict(list)

# Define your checks and maps
SIDE_CHECKS = {'label', 'botle_with_neckband', 'curved_shoulder'}
TOP_CHECKS = {'Cap'}
BOTTOM_CHECKS = {'Base'}

SIDE_CHECKS_MAP = {
    'label': ('Label', 'Present'),
    'botle_with_neckband': ('Neckband', 'Present'),
    'curved_shoulder': ('Shoulder', 'Curved')
}

def update_CHECKLIST(key, value, checklist):
    checklist[key].append(value)

