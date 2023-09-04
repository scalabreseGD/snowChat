create_config() {
  if [ -e "$REDASH_BASE_PATH"/env ]; then
    rm "$REDASH_BASE_PATH"/env
    touch "$REDASH_BASE_PATH"/env
  fi

  COOKIE_SECRET=$(pwgen -1s 32)
  SECRET_KEY=$(pwgen -1s 32)

  cat <<EOF >"$PWD"/env
PYTHONUNBUFFERED=0
REDASH_LOG_LEVEL=INFO
REDASH_COOKIE_SECRET=$COOKIE_SECRET
REDASH_SECRET_KEY=$SECRET_KEY
EOF
}

#setup_compose() {
#  REQUESTED_CHANNEL=stable
#  LATEST_VERSION=$(curl -s "https://version.redash.io/api/releases?channel=$REQUESTED_CHANNEL" | json_pp | grep "docker_image" | head -n 1 | awk 'BEGIN{FS=":"}{print $3}' | awk 'BEGIN{FS="\""}{print $1}')
#
#  cd "$REDASH_BASE_PATH"
#  GIT_BRANCH="${REDASH_BRANCH:-master}" # Default branch/version to master if not specified in REDASH_BRANCH env var
#  curl -OL https://raw.githubusercontent.com/getredash/setup/"$GIT_BRANCH"/data/docker-compose.yml
#  sed -ri "s/image: redash\/redash:([A-Za-z0-9.-]*)/image: redash\/redash:$LATEST_VERSION/" docker-compose.yml
#  echo "export COMPOSE_PROJECT_NAME=redash" >>~/.profile
#  echo "export COMPOSE_FILE=$REDASH_BASE_PATH/docker-compose.yml" >>~/.profile
#  export COMPOSE_PROJECT_NAME=redash
#  export COMPOSE_FILE="$REDASH_BASE_PATH"/docker-compose.yml
#  sudo docker-compose run --rm server create_db
#  sudo docker-compose up -d
#}

create_config
#setup_compose
