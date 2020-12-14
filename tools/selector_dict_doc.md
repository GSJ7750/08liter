* collection
    1. [div] sort_button_area
        * [xpath] sort_button
            - [a] sort_button_click_wait()
    2.[div] reviews_section #리뷰파트 전체를 감싸고 있는 div
        * [ul] review_list #실제 리뷰를 담고있는 리스트
            * [p] review_text
            * [span] review_info_area
            * [span] review_score
    3. [div] review_page_button_area
    * [p] err_no_product #존재하지 않는 상품
    * [div] err_sale_stopped #판매 중단 및 리뷰가 삭제된 상품
* coupang
    1. [li] move_to_review_button #tab메뉴의 상품평 버튼
        - [li] move_to_review_button_click_wait() #fixed tab으로 변환될 때
    2. [div] sort_button_area
        * sort_button #find_by_class_name method로 검색하기 때문에 tag가 없음
        * [button] sort_button_click_wait #최신순 정렬이 완료됐을 때
    3. [article] review_cell #각각의 리뷰를 지칭
        * [div] review_title 
        * [div] review_text
        * [div] review_score
        * [div] review_date
    4. [div] review_page_button_area
        * [button] review_next_page_button
        * [button] review_select_page_button #특정 페이지로 바로 넘어감
        - [button] review_page_change_wait()
* smartstore
    1. [div] sort_button_area
        * [xpath] sort_button
            - [xpath] sort_button_click_wait()
    2. [div] review_list
        * [li] review_cell
            * [em] review_score
            * [div] review_info_area
            * [div] review_text_area
                * [span] review_text
    3. [div] review_page_button_area
        * [a] review_next_page_button
            - [attribute] is_last_page(= aria_hidden)
        * [a] review_select_page_button
        - [a] review_page_change_wait
    * [strong] err_blocked #네이버 막힘
    * [div] err_sale_stopped #판매 중지된 상품
    * [div] err_closed_market #운영 중지된 쇼핑몰
    * [span] err_review_count_zero #리뷰가 보이지 않는 오류