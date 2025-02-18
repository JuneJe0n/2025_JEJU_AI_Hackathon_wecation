import streamlit as st
from dotenv import load_dotenv

from program_for_user import rank_programs_for_user
from embedding import embed_users, embed_programs
from matching import dbscan_clustering, match_teams
from util import load_data, get_fitness

load_dotenv()

st.title("✈️ Jeju-Wecation Demo 🍊")

if "program" not in st.session_state:
    from embedding import embed_programs
    st.session_state.program_info = load_data("dataset/program_dummy_data.json")
    st.session_state.program_embed = embed_programs(st.session_state.program_info)
    st.session_state.program = True

if "user" not in st.session_state:
    from embedding import embed_users
    st.session_state.user_info = load_data("dataset/user_dummy_data.json")
    st.session_state.user_embed = embed_users(st.session_state.user_info)
    st.session_state.user = True

if "page1" not in st.session_state:
    st.session_state.page1 = True
if "page2" not in st.session_state:
    st.session_state.page2 = False
if "page3" not in st.session_state:
    st.session_state.page3 = False

if st.session_state.page1:
    with st.form(key="my_info"):
        st.subheader("내 기본 정보")
        col11, col12, col13 = st.columns(3)

        with col11:
            name = st.text_input("이름", key="my_info_name")
            sex = st.radio("성별", ("남", "여"), horizontal=True, key="my_info_sex")
            phone = st.text_input("전화번호", key="my_info_phone")

        with col12:
            age = st.text_input("나이", key="my_info_age")
            job = st.selectbox("직업군", ("개발자", "디자이너", "회계/법"), key="my_info_job")
            mbti = st.selectbox("MBTI", ("ENFP", "ISFP"), key="my_info_mbti")

        with col13:
            introduction = st.text_area("간단한 자기소개", height=205, key="my_info_introduction")

        st.divider()
        col21, col22, col23 = st.columns(3)

        with col21:
            interest = st.pills("당신의 관심사를 선택해주세요.", ("아웃도어 활동", "문화 예술", "문화 탐방", "비즈니스 네트워킹"), selection_mode="multi", key="my_info_interest")
            person = st.radio("매칭그룹 인원", ("3명", "4명", "5명", "6명"), horizontal=True, key="my_info_person")

        with col22:
            location = st.radio("가고싶은 여행지", ("한라산", "제주 시내", "제주 서부", "제주 동부", "서귀포/중문", "그 외 권역"), horizontal=False, key="my_info_location")

        with col23:
            date = st.date_input("네트워킹 원하는 날짜를 선택해주세요.", key="my_info_data")

        col31, col32 = st.columns([9,1])
        
        with col32:
            button = st.form_submit_button("저장")
            
            if button:
                st.session_state.info = {
                    "basic_info": {
                        "user_id": "0",
                        "name": name,
                        "sex": sex,
                        "age": age,
                        "job": job,
                        "exp": introduction
                    },
                    "added_info": {
                        "interest": interest,
                        "region": location,
                        "date": date
                    }
                }
                st.session_state.person = int(person.rstrip("명"))

    if "info" in st.session_state:
        col1, col2 = st.columns(2)

        with col1:
            match_button = st.button("다른 사람들과 AI 매칭 시작하기", use_container_width=True, key="match_button")
            if match_button:
                st.session_state.page1 = False
                st.session_state.page2 = True
                st.rerun()

        with col2:
            recommendation_button = st.button("AI로 프로그램 추천받기", use_container_width=True, key="recommendation_button")
            if recommendation_button:
                st.session_state.page1 = False
                st.session_state.page3 = True
                st.rerun()

if st.session_state.page2:
    st.subheader("당신과 매칭된 파트너들")

    my_embed = embed_users([st.session_state.info])
    for k, v in my_embed.items():
        st.session_state.user_embed[k] = v

    clusters, outliers = dbscan_clustering(st.session_state.user_embed)
    team_db, additional_outliers = match_teams(clusters, st.session_state.user_embed, st.session_state.program_embed)

    for k, v in team_db.items():
        if st.session_state.info["basic_info"]["user_id"] in v["users"]:
            topk_user_id = v["users"]
            topk_user_id.remove(st.session_state.info["basic_info"]["user_id"])
            break
    
    topk_users = []
    for id in topk_user_id:
        for user in st.session_state.user_info:
            if user["basic_info"]["user_id"] == id:
                topk_users.append(user)
                break
    
    def profile(user):
        with st.container(border=True):
            st.markdown(f"이름: {user['basic_info']['name']}")
            st.markdown(f"성별: {user['basic_info']['sex']}")
            st.markdown(f"나이: {user['basic_info']['age']}")
            st.markdown(f"직업군: {user['basic_info']['job']}")


    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        idx = 0
        with col1:
            profile(topk_users[idx])
            idx += 1
        with col2:
            profile(topk_users[idx])
            idx += 1
        with col3:
            profile(topk_users[idx])
            idx += 1
        with col4:
            profile(topk_users[idx])
            idx += 1

if st.session_state.page3:
    topk_id = rank_programs_for_user(st.session_state.info["basic_info"]["user_id"], [st.session_state.info], st.session_state.program_info)[:6]
    topk_programs = []
    for id in topk_id:
        for program in st.session_state.program_info:
            if program["program_id"] == id:
                topk_programs.append(program)
                break
    
    my_embed = embed_users([st.session_state.info])
    topk_program_embed = embed_programs(topk_programs)
    fitnesses = get_fitness(my_embed, topk_program_embed)
    for i in range(len(topk_programs)):
        topk_programs[i]["fitness"] = fitnesses[0][i]
        
    def show_programs(programs):
        idx = 0
        for _ in range((len(programs) // 3) + 1):
            if idx == len(programs):
                break
            col1, col2, col3 = st.columns(3)
            with col1:
                program_box(programs[idx])
                idx += 1
                if idx == len(programs):
                    break
            with col2: 
                program_box(programs[idx])
                idx += 1
                if idx == len(programs):
                    break
            with col3:
                program_box(programs[idx])
                idx += 1
                if idx == len(programs):
                    break
            
    tab1, tab2 = st.tabs(["전체 프로그램", "추천 프로그램"])

    with tab1:
        def program_box(program):
            with st.container(border=True):
                st.markdown(f"{program['name']}")
                st.caption(f"{program['info']}")
                st.caption(f"{'   '.join(program['hashtag'])}")
                st.caption(f"모집 현황: {program['person']}/{program['limit']} ({program['ongoing']})")

        show_programs(st.session_state.program_info)
    
    with tab2:
        def program_box(program):
            with st.container(border=True):
                st.markdown(f"{program['name']}")
                st.caption(f"{program['info']}")
                st.caption(f"{'   '.join(program['hashtag'])}")
                st.caption(f"모집 현황: {program['person']}/{program['limit']} ({program['ongoing']})")
                st.caption(f"추천률: {program['fitness']}%")
        
        show_programs(topk_programs)
    
    



