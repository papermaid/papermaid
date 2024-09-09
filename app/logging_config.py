import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} | {module} | {message}',
            'style': '{'
        },
        'simple': {
            'format': '{levelname} | {message}',
            'style': '{'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',  # Shows DEBUG and higher level messages in the console
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'file': {
            'level': 'INFO',  # Shows INFO and higher level messages in the file
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'formatter': 'verbose'
        },
    },
    'loggers': {
        'papermaid': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG',  # Minimum level of messages to capture
            'propagate': False,
        },
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
