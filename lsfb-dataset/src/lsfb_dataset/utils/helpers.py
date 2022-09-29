
def duration_to_str(duration):
    milli = duration % 1000
    seconds = (duration // 1000) % 60
    minutes = duration // 60000
    return f'{minutes}min {seconds:2}s {milli:3}ms'
