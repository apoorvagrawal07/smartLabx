# Kafka Consumers

Kafka consumers read records from partitions.

Each partition is consumed by only one consumer
within a consumer group at a time.

If a consumer crashes, Kafka detects the failure
through the consumer group's coordination mechanism
and reassigns the affected partitions to another
consumer in the group.

Consumers periodically commit offsets so that they
can resume processing from a known position after
a restart.